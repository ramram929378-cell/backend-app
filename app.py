import os
import uuid
import aiofiles
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from sqlmodel import SQLModel, Field, create_engine, Session, select

from groq import Groq
import replicate
from twilio.rest import Client as TwilioClient

# ---------- DB Model ----------
class UserMemory(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str
    key: str
    value: str

# ---------- INIT ----------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)

app = FastAPI()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
replicate_client = replicate
twilio_client = TwilioClient(
    os.getenv("TWILIO_SID"),
    os.getenv("TWILIO_TOKEN")
)

# ---------- CHAT ----------
@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    message: str = Form(...),
    persona: str = Form("hero")
):
    system_prompt = (
        "You are a caring Telugu mass-hero style companion. "
        "Emotional, powerful dialogue style. Persona: " + persona
    )

    response = groq_client.chat.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        stream=False
    )

    reply = response.choices[0].message["content"]
    return {"reply": reply}

# ---------- SPEECH TO TEXT ----------
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    file_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(file_path, "wb") as out:
        content = await file.read()
        await out.write(content)

    with open(file_path, "rb") as f:
        text_result = groq_client.speech.transcribe(
            f,
            model="whisper-large-v3"
        )

    return {"text": text_result["text"]}

# ---------- TEXT TO SPEECH ----------
@app.post("/tts")
async def tts(
    text: str = Form(...),
    voice: str = Form("telugu_male_1")
):
    audio_bytes = groq_client.speech.synthesize(
        text=text,
        voice=voice,
        format="mp3"
    )

    out_path = f"/tmp/{uuid.uuid4()}.mp3"
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    return FileResponse(out_path, media_type="audio/mpeg", filename="reply.mp3")

# ---------- IMAGE GENERATION ----------
@app.post("/generate_image")
async def generate_image(
    prompt: str = Form(...),
    style: str = Form(None)
):
    output = replicate.run(
        "stability-ai/stable-diffusion-2",
        input={"prompt": prompt}
    )
    return {"images": output}

# ---------- MEMORY WRITE ----------
@app.post("/memory")
def write_memory(
    user_id: str = Form(...),
    key: str = Form(...),
    value: str = Form(...)
):
    with Session(engine) as s:
        m = UserMemory(user_id=user_id, key=key, value=value)
        s.add(m)
        s.commit()
        return {"ok": True}

# ---------- MEMORY READ ----------
@app.get("/memory/{user_id}")
def read_memory(user_id: str):
    with Session(engine) as s:
        stmt = select(UserMemory).where(UserMemory.user_id == user_id)
        res = s.exec(stmt).all()
        return {"memory": [{r.key: r.value} for r in res]}

# ---------- OUTBOUND CALL ----------
@app.post("/call")
def make_call(
    to_number: str = Form(...),
    tts_text: str = Form(...)
):
    call = twilio_client.calls.create(
        twiml=f'<Response><Say language="en-IN">{tts_text}</Say></Response>',
        to=to_number,
        from_=os.getenv("TWILIO_FROM")
    )
    return {"sid": call.sid}
