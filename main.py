"""
Escritorio Digital IA - Backend Principal
"""
import os, json, logging
from datetime import datetime
from typing import Optional
import anthropic, requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
PINECONE_KEY   = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "jurisprudencia-br")
CLICKSIGN_KEY  = os.getenv("CLICKSIGN_API_KEY", "")
DATAJUD_KEY    = os.getenv("DATAJUD_API_KEY", "")

claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

app = FastAPI(title="Escritorio Digital IA", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class LeadInput(BaseModel):
    descricao: str
    documentos_ocr: Optional[str] = ""
    nome: Optional[str] = ""
    telefone: Optional[str] = ""

class ContratoInput(BaseModel):
    nome: str
    cpf: str
    email: str
    telefone: str
    tese: str
    valor_min: float = 0
    valor_max: float = 0
    info_extra: Optional[str] = ""

class PeticaoInput(BaseModel):
    nome: str
    cpf: str
    tese: str
    fatos: str
    reu: str
    valor_principal: float = 0
    dano_moral: float = 0

class WppInput(BaseModel):
    mensagem: str
    nome_cliente: Optional[str] = ""
    fase_processo: Optional[str] = "em andamento"
    numero_processo: Optional[str] = ""
    historico: Optional[list] = []

PROMPT_CLASSIFICADOR = '''Voce eh um assistente juridico especializado em triagem de casos brasileiros.
Classifique o caso e retorne APENAS JSON valido:
{
  "tese": "rmc_rcc|trabalhista|auxilio_acidente|plano_saude|ir_doenca_grave|iptu|itbi|sabesp_fator_k",
  "tese_label": "nome amigavel",
  "viabilidade": 0,
  "viabilidade_justificativa": "1-2 frases",
  "documentos_faltantes": [],
  "documentos_criticos_faltantes": false,
  "prescrito": false,
  "valor_estimado_min": 0.0,
  "valor_estimado_max": 0.0,
  "ticket_honorarios_estimado": 0.0,
  "urgencia": "alta|media|baixa",
  "recomendacao": "aceitar|recusar|solicitar_docs",
  "notas_advogado": "observacoes - para rmc_rcc mencionar Tema 1328 STJ sobrestado",
  "mensagem_cliente": "mensagem curta para whatsapp"
}
Ticket = 30 porcento do valor medio.'''

PROMPT_CONTRATO = '''Voce eh advogado especialista em contratos de honorarios advocaticios no Brasil.
Gere contrato de honorarios de exito (sem pagamento antecipado), honorarios 30 porcento do valor liquido.
Base legal: Art. 22 Lei 8.906/94. Valido para assinatura eletronica (Lei 14.063/2020).
Inclua: qualificacao das partes, objeto, obrigacoes, honorarios, procuracao ad judicia et extra, foro.
Linguagem clara e acessivel.'''

PROMPT_PETICAO = '''Voce eh advogado especialista em advocacia de massa brasileira.
Gere peticoes iniciais completas, tecnicas e persuasivas.
REGRAS:
- Estrutura completa: enderecamento, qualificacao, fatos, direito, pedidos numerados, valor da causa
- Para rmc_rcc: art. 42 CDC, Sumula 297 STJ, IN INSS 28/2008, Tema 929 STJ (dobro pos 30/03/2021)
  Tema 1328 STJ sobrestado: dano moral APENAS como pedido subsidiario
- Tutela de urgencia quando cabivel
- JEC se valor menor ou igual a R$ 56480 (40 SM), Vara Civel acima
- Citar jurisprudencia do STJ e TJSP'''

PROMPT_JULIA = '''Voce eh Julia, assistente juridica virtual do escritorio de advocacia digital.
Portugues coloquial e respeitoso. Mensagens curtas (max 3 paragrafos). Emojis com moderacao.
ESCALE IMEDIATAMENTE (responda ESCALAR: motivo) se o cliente mencionar:
intimacao, citacao, audiencia, quero cancelar, recurso, reclamacao OAB, falar com advogado
Nunca garanta prazo ou resultado.'''

TESE_LABELS = {
    "rmc_rcc": "declaracao de nulidade de contrato RMC/RCC e repeticao de indebito em dobro",
    "trabalhista": "reclamatoria trabalhista - verbas rescisorias e direitos trabalhistas",
    "auxilio_acidente": "concessao/restituicao de auxilio-acidente B36 junto ao INSS",
    "plano_saude": "acao por negativa de cobertura de plano de saude",
    "ir_doenca_grave": "isencao de IRPF sobre aposentadoria por doenca grave (Lei 7.713/88)",
    "iptu": "revisao de IPTU e restituicao de valores cobrados indevidamente",
    "itbi": "restituicao de ITBI cobrado em excesso",
    "sabesp_fator_k": "nulidade da cobranca do Fator K e restituicao - SABESP",
}

JURIS_FALLBACK = {
    "rmc_rcc": "[STJ Tema 929] EAREsp 676.608/RS - repeticao em dobro pos 30/03/2021\n[STJ Tema 1328] REsp 2.145.244/SC - dano moral in re ipsa SOBRESTADO\n[TJSP] AC 1016333-05.2020 - conversao RMC para emprestimo consignado\n[STJ] Sumula 297 - CDC aplicavel as instituicoes financeiras",
    "trabalhista": "[TST] Sumula 338 - onus do empregador registrar jornada\n[TST] Sumula 85 - banco de horas exige acordo coletivo",
    "sabesp_fator_k": "[TJSP] Sumula 193 - SABESP nao pode cobrar Fator K sem prestacao efetiva\n[TJSP] Prescricao 5 anos - art. 27 CDC",
    "ir_doenca_grave": "[STJ] REsp 1.116.620/BA - isencao IR sobre proventos de portador de doenca grave\n[TRF3] Cardiopatia grave CID I25 qualificada para isencao Lei 7.713/88",
}

def chamar_claude(system, user, model="claude-sonnet-4-20250514", max_tokens=4000):
    if not claude:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY nao configurada no .env")
    resp = claude.messages.create(model=model, max_tokens=max_tokens, system=system,
                                   messages=[{"role": "user", "content": user}])
    return resp.content[0].text

def buscar_juris(tese, fatos):
    if not (PINECONE_KEY and OPENAI_KEY):
        return JURIS_FALLBACK.get(tese, "[STJ] Sumula 297 - CDC aplicavel as instituicoes financeiras")
    try:
        from openai import OpenAI
        from pinecone import Pinecone
        oa = OpenAI(api_key=OPENAI_KEY)
        pc = Pinecone(api_key=PINECONE_KEY)
        idx = pc.Index(PINECONE_INDEX)
        emb = oa.embeddings.create(model="text-embedding-3-large", input=fatos[:6000], dimensions=3072).data[0].embedding
        res = idx.query(vector=emb, top_k=5, filter={"tese_interna": {"$eq": tese}}, include_metadata=True)
        linhas = []
        for m in res["matches"]:
            if m["score"] < 0.72: continue
            meta = m["metadata"]
            linhas.append(f"[{meta.get('tribunal')}] {meta.get('numero_processo')} | {meta.get('data_julgamento')} | {meta.get('resultado','').upper()}\n{meta.get('ementa','')[:300]}")
        return "\n\n".join(linhas) if linhas else JURIS_FALLBACK.get(tese, "")
    except Exception as e:
        log.warning(f"Pinecone fallback: {e}")
        return JURIS_FALLBACK.get(tese, "")

def enviar_clicksign(texto, nome, email, cpf):
    if not CLICKSIGN_KEY:
        return {"status": "simulado", "link": f"https://app.clicksign.com/sign/DEMO-{cpf[-4:]}", "doc_key": "DEMO"}
    import base64
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    base_url = "https://app.clicksign.com/api/v1"
    doc_b64 = base64.b64encode(texto.encode()).decode()
    r1 = requests.post(f"{base_url}/documents?access_token={CLICKSIGN_KEY}", headers=headers, json={
        "document": {"path": f"/contratos/{cpf.replace('.','').replace('-','')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt",
                     "content_base64": f"data:text/plain;base64,{doc_b64}", "auto_close": True, "locale": "pt-BR"}})
    r1.raise_for_status()
    doc_key = r1.json()["document"]["key"]
    r2 = requests.post(f"{base_url}/signers?access_token={CLICKSIGN_KEY}", headers=headers, json={
        "signer": {"email": email, "name": nome, "documentation": cpf, "has_documentation": True}})
    r2.raise_for_status()
    signer_key = r2.json()["signer"]["key"]
    r3 = requests.post(f"{base_url}/lists?access_token={CLICKSIGN_KEY}", headers=headers, json={
        "list": {"document_key": doc_key, "signer_key": signer_key, "sign_as": "party",
                 "message": f"Ola {nome.split()[0]}! Seu contrato esta pronto. Clique para assinar!"}})
    r3.raise_for_status()
    return {"status": "enviado", "link": r3.json()["list"]["url"], "doc_key": doc_key}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat(),
            "servicos": {
                "claude": "ativo" if ANTHROPIC_KEY else "sem chave",
                "openai": "ativo" if OPENAI_KEY else "nao configurado",
                "pinecone": "ativo" if PINECONE_KEY else "nao configurado (usando fallback)",
                "clicksign": "ativo" if CLICKSIGN_KEY else "nao configurado (links simulados)",
                "datajud": "ativo" if DATAJUD_KEY else "nao configurado",
            }}

@app.post("/classificar")
async def classificar(body: LeadInput):
    user = f"## DESCRICAO\n{body.descricao}"
    if body.documentos_ocr:
        user += f"\n\n## DOCUMENTOS (OCR)\n{body.documentos_ocr}"
    raw = chamar_claude(PROMPT_CLASSIFICADOR, user, model="claude-haiku-4-5-20251001", max_tokens=1000)
    try:
        return json.loads(raw.replace("```json","").replace("```","").strip())
    except Exception:
        raise HTTPException(status_code=500, detail=f"Erro ao parsear JSON: {raw[:200]}")

@app.post("/contrato")
async def gerar_contrato(body: ContratoInput):
    user = f"""Gere contrato com estes dados:
ESCRITORIO: Escritorio Digital Advocacia
CLIENTE: {body.nome} | CPF: {body.cpf} | Email: {body.email} | Tel: {body.telefone}
OBJETO: {TESE_LABELS.get(body.tese, body.tese)}
VALOR ESTIMADO: R$ {body.valor_min:,.0f} a R$ {body.valor_max:,.0f}
HONORARIOS: 30 porcento do valor liquido obtido (exito puro)
{f'INFO: {body.info_extra}' if body.info_extra else ''}
DATA: {datetime.now().strftime('%d/%m/%Y')}"""
    texto = chamar_claude(PROMPT_CONTRATO, user, max_tokens=3000)
    cs = enviar_clicksign(texto, body.nome, body.email, body.cpf)
    return {"contrato_texto": texto, "clicksign_link": cs["link"], "clicksign_status": cs["status"],
            "clicksign_doc_key": cs["doc_key"],
            "mensagem_wpp": f"Ola {body.nome.split()[0]}! Seu contrato esta pronto.\n\nClique para assinar:\n{cs['link']}\n\nAssim que assinar, nossa equipe comeca a trabalhar!"}

@app.post("/peticao")
async def gerar_peticao(body: PeticaoInput):
    valor_total = body.valor_principal + body.dano_moral
    juizo = "JEC (Juizado Especial Civel)" if valor_total <= 56480 else "Vara Civel"
    protocolo = {"auxilio_acidente": "SEEU", "trabalhista": "PJe TRT"}.get(body.tese, "eSAJ / PJe")
    juris = buscar_juris(body.tese, body.fatos)
    user = f"""Gere a peticao inicial:
REQUERENTE: {body.nome} | CPF: {body.cpf}
REU: {body.reu}
TESE: {body.tese} - {TESE_LABELS.get(body.tese, body.tese)}
JUIZO: {juizo} | PROTOCOLO: {protocolo}
VALOR PRINCIPAL: R$ {body.valor_principal:,.2f}
DANO MORAL (subsidiario): R$ {body.dano_moral:,.2f}
VALOR TOTAL: R$ {valor_total:,.2f}
FATOS: {body.fatos}
JURISPRUDENCIA DISPONIVEL (cite com tribunal, numero e data):
{juris}
Gere a peticao inicial completa agora."""
    texto = chamar_claude(PROMPT_PETICAO, user, max_tokens=6000)
    return {"peticao_texto": texto, "juizo": juizo, "protocolo": protocolo,
            "valor_causa": valor_total, "rag_ativo": bool(PINECONE_KEY and OPENAI_KEY)}

@app.post("/wpp")
async def atendimento_wpp(body: WppInput):
    system = f"""{PROMPT_JULIA}
CLIENTE: {body.nome_cliente or 'Cliente'}
PROCESSO: {body.numero_processo or 'em tramitacao'}
FASE: {body.fase_processo}"""
    msgs = (body.historico or [])[-10:]
    msgs.append({"role": "user", "content": body.mensagem})
    resp = claude.messages.create(model="claude-haiku-4-5-20251001", max_tokens=400,
                                   system=system, messages=msgs).content[0].text
    escalar = resp.upper().startswith("ESCALAR:") or any(
        g in body.mensagem.lower() for g in ["cancelar","desistir","intimacao","citacao","audiencia","recurso","oab"])
    motivo = None
    if resp.upper().startswith("ESCALAR:"):
        motivo = resp.split(":",1)[1].strip().split("\n")[0]
        resp = "Entendi! Vou transferir voce agora para um de nossos advogados. Um momento!"
    return {"resposta": resp, "escalar": escalar, "motivo_escalonamento": motivo}
