import os, ast, json, sqlite3, operator
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document as DocxDocument

load_dotenv()
st.set_page_config(page_title="StudyMate AI", page_icon="🎓", layout="wide")
st.title("🎓 StudyMate AI")
st.caption("Your personal AI study assistant — chat with documents, create quizzes and organize study tasks.")
api_key=os.getenv("OPENAI_API_KEY","")
try: api_key=st.secrets.get("OPENAI_API_KEY",api_key)
except Exception: pass
if not api_key:
    st.warning("Add OPENAI_API_KEY to .env locally or Streamlit App Secrets.")
    st.code('OPENAI_API_KEY = "your_key_here"')
    st.stop()
client=OpenAI(api_key=api_key)
model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
embed_model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small")
DB="studymate_tasks.db"

def conn_db():
    c=sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, task TEXT, due TEXT DEFAULT '', status TEXT DEFAULT 'Pending')")
    c.commit()
    return c

def add_task(task,due=""):
    c=conn_db(); cur=c.execute("INSERT INTO tasks(task,due) VALUES(?,?)",(task,due)); c.commit()
    out={"success":True,"id":cur.lastrowid,"task":task,"due":due}; c.close(); return out

def list_tasks():
    c=conn_db(); rows=c.execute("SELECT id,task,due,status FROM tasks ORDER BY id DESC").fetchall(); c.close()
    return [{"id":r[0],"task":r[1],"due":r[2],"status":r[3]} for r in rows]

def mark_done(task_id):
    c=conn_db(); cur=c.execute("UPDATE tasks SET status='Completed' WHERE id=?",(task_id,)); c.commit(); n=cur.rowcount; c.close()
    return {"success":n>0,"id":task_id}

def extract(f):
    n=f.name.lower()
    if n.endswith(".pdf"): return "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
    if n.endswith(".docx"): return "\n".join(p.text for p in DocxDocument(f).paragraphs)
    if n.endswith(".txt"): return f.getvalue().decode("utf-8",errors="ignore")
    return ""

def chunks_of(text,size=180,overlap=35):
    w=text.split(); out=[]; start=0
    while start<len(w):
        end=min(start+size,len(w)); out.append(" ".join(w[start:end]))
        if end==len(w): break
        start=end-overlap
    return out

def embed(texts):
    result=[]
    for i in range(0,len(texts),64):
        r=client.embeddings.create(model=embed_model,input=texts[i:i+64])
        result.extend(x.embedding for x in r.data)
    a=np.asarray(result,dtype="float32")
    return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)

def retrieve(q,docs,vectors,k=4):
    if not docs: return []
    v=client.embeddings.create(model=embed_model,input=[q]).data[0].embedding
    v=np.asarray(v,dtype="float32"); v=v/max(np.linalg.norm(v),1e-12)
    scores=vectors@v
    ids=np.argsort(scores)[::-1][:min(k,len(docs))]
    return [{"score":float(scores[i]),**docs[int(i)]} for i in ids]

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow,ast.USub:operator.neg,ast.UAdd:operator.pos}
def calc(expr):
    def walk(n):
        if isinstance(n,ast.Expression): return walk(n.body)
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:
            a,b=walk(n.left),walk(n.right)
            if isinstance(n.op,ast.Pow) and abs(b)>8: raise ValueError("Exponent too large")
            return OPS[type(n.op)](a,b)
        if isinstance(n,ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](walk(n.operand))
        raise ValueError("Only basic arithmetic is allowed")
    if len(expr)>100: raise ValueError("Expression too long")
    return walk(ast.parse(expr,mode="eval"))

TOOLS=[
{"type":"function","function":{"name":"calculate","description":"Calculate basic arithmetic.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"],"additionalProperties":False}}},
{"type":"function","function":{"name":"add_study_task","description":"Add a task only when the user explicitly asks.","parameters":{"type":"object","properties":{"task":{"type":"string"},"due":{"type":"string"}},"required":["task"],"additionalProperties":False}}},
{"type":"function","function":{"name":"list_study_tasks","description":"List saved study tasks.","parameters":{"type":"object","properties":{},"additionalProperties":False}}}
]
def tool(name,args):
    if name=="calculate": return {"result":calc(args["expression"])}
    if name=="add_study_task": return add_task(args["task"],args.get("due",""))
    if name=="list_study_tasks": return {"tasks":list_tasks()}
    return {"error":"Unknown tool"}

def ask_agent(prompt,context=""):
    sys=("You are StudyMate AI, a friendly study assistant. Use tools for calculations and task operations. "
         "Only save tasks when explicitly asked. Never claim an action succeeded unless confirmed by tool result. "
         "When document context is provided, answer from it and say if the answer is missing.")
    msgs=[{"role":"system","content":sys}]
    if context: msgs.append({"role":"system","content":"Retrieved document passages:\n"+context})
    msgs.append({"role":"user","content":prompt}); logs=[]
    for _ in range(5):
        r=client.chat.completions.create(model=model,messages=msgs,tools=TOOLS,tool_choice="auto",temperature=.2)
        m=r.choices[0].message
        if not m.tool_calls: return m.content or "",logs
        msgs.append(m)
        for call in m.tool_calls:
            try: args=json.loads(call.function.arguments or "{}"); result=tool(call.function.name,args)
            except Exception as e: args={}; result={"error":str(e)}
            logs.append({"tool":call.function.name,"arguments":args,"result":result})
            msgs.append({"role":"tool","tool_call_id":call.id,"content":json.dumps(result)})
    return "Tool-call limit reached; please simplify your request.",logs

with st.sidebar:
    st.subheader("📅 Study tasks")
    for t in list_tasks():
        st.write(f"{'✅' if t['status']=='Completed' else '🕒'} **#{t['id']}** {t['task']}")
        if t["due"]: st.caption("Due: "+t["due"])
        if t["status"]!="Completed" and st.button("Mark complete",key=f"done{t['id']}"):
            mark_done(t["id"]); st.rerun()

chat,quiz,tasks,about=st.tabs(["💬 AI Study Chat","📝 Quiz Generator","📅 Task Planner","ℹ️ About"])
with chat:
    st.subheader("Chat with your documents")
    files=st.file_uploader("Upload PDF, DOCX or TXT",type=["pdf","docx","txt"],accept_multiple_files=True)
    if files and st.button("Process documents"):
        with st.spinner("Extracting text and generating embeddings..."):
            try:
                docs=[]
                for f in files:
                    docs += [{"source":f.name,"text":x} for x in chunks_of(extract(f)) if x.strip()]
                vectors=embed([d["text"] for d in docs]) if docs else None
                st.session_state.docs=docs; st.session_state.vectors=vectors
                st.success(f"Processed {len(files)} file(s), {len(docs)} chunks.")
            except Exception as e: st.error(f"Processing error: {e}")
    docs=st.session_state.get("docs",[]); vectors=st.session_state.get("vectors")
    if docs: st.caption(f"Knowledge base ready: {len(docs)} chunks")
    if "history" not in st.session_state: st.session_state.history=[]
    for m in st.session_state.history:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    prompt=st.chat_input("Ask a question, calculate, or add a study task...")
    if prompt:
        st.session_state.history.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                hits=retrieve(prompt,docs,vectors) if docs else []
                context="\n\n".join(f"Source: {x['source']}\n{x['text']}" for x in hits)
                answer,logs=ask_agent(prompt,context); st.markdown(answer)
                if hits:
                    with st.expander("Retrieved document passages"):
                        for h in hits: st.markdown(f"**{h['source']} · {h['score']:.3f}**\n\n{h['text']}")
                if logs:
                    with st.expander("Agent tool calls"):
                        for x in logs: st.json(x)
        st.session_state.history.append({"role":"assistant","content":answer})

with quiz:
    st.subheader("Generate a practice quiz")
    topic=st.text_input("Topic or chapter",placeholder="e.g. RAG, photosynthesis, Python functions")
    level=st.selectbox("Difficulty",["Beginner","Intermediate","Advanced"])
    count=st.slider("Questions",3,10,5)
    if st.button("Generate quiz",disabled=not topic.strip()):
        context=""
        if st.session_state.get("docs"):
            hits=retrieve(topic,st.session_state.docs,st.session_state.vectors,6)
            context="\n\n".join(x["text"] for x in hits)
        try:
            r=client.chat.completions.create(model=model,temperature=.3,response_format={"type":"json_object"},
             messages=[{"role":"system","content":"Return valid JSON only: {title:string, questions:[{question:string, options:[4 strings], answer:string, explanation:string}]}. Make exactly the requested number. Answer must exactly match one option."},
             {"role":"user","content":f"Create {count} {level} MCQs on {topic}. Use this context if present:\n{context}"}])
            st.session_state.quiz=json.loads(r.choices[0].message.content)
        except Exception as e: st.error(f"Quiz generation error: {e}")
    qz=st.session_state.get("quiz")
    if qz:
        st.markdown("### "+qz.get("title","Practice quiz"))
        with st.form("quizform"):
            ans={}
            for i,q in enumerate(qz.get("questions",[])):
                st.markdown(f"**Q{i+1}. {q['question']}**")
                ans[i]=st.radio("Select answer",q["options"],key=f"q{i}",index=None,label_visibility="collapsed")
            submitted=st.form_submit_button("Check answers")
        if submitted:
            score=0
            for i,q in enumerate(qz.get("questions",[])):
                ok=ans[i]==q["answer"]; score+=ok
                st.write(("✅" if ok else "❌")+f" Q{i+1}: {q['answer']}")
                st.caption(q.get("explanation",""))
            st.success(f"Score: {score}/{len(qz.get('questions',[]))}")

with tasks:
    st.subheader("Study task planner")
    with st.form("taskform",clear_on_submit=True):
        t=st.text_input("Task",placeholder="Revise Module 04")
        d=st.text_input("Due date (optional)",placeholder="2026-11-10")
        submit=st.form_submit_button("Add task")
        if submit and t.strip(): add_task(t.strip(),d.strip()); st.success("Task added."); st.rerun()
    st.dataframe(list_tasks(),use_container_width=True,hide_index=True)

with about:
    st.markdown("""
    **StudyMate AI integrates:**
    - LLM API for chat and quiz generation
    - RAG: document extraction → chunking → embeddings → similarity retrieval → grounded answer
    - Structured JSON output for quizzes
    - Function calling and agent loop for calculator and task operations
    - SQLite task storage and Streamlit session state

    **Limitations:** generated answers and quiz keys may be wrong; verify them. Scanned PDFs may need OCR. Similarity is not a confidence score. Uploaded text is sent to the configured embedding API. Hosted SQLite storage may be temporary.
    """)
