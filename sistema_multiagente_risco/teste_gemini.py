
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import os

# Carrega as variáveis do .env desta pasta
pasta_projeto = Path(__file__).resolve().parent
load_dotenv(pasta_projeto / ".env")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY não encontrada no arquivo .env."
    )

cliente = genai.Client(api_key=api_key)

resposta = cliente.models.generate_content(
    model="gemini-3.8-flash",
    contents="Responda apenas: Conexão Gemini OK."
)

print("Resposta do Gemini:")
print(resposta.text)
