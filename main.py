"""Escritorio Digital IA - Backend v2"""
import os, json, logging, base64
from datetime import datetime
from typing import Optional
import anthropic, requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "jurisprudencia-br")
CLICKSIGN_KEY = os.getenv("CLICKSIGN_API_KEY", "")
DATAJUD_KEY = os.getenv("DATAJUD_API_KEY", "")
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None
app = FastAPI(title="Escritorio Digital IA", version="2.0.0")
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

class ExtrairInput(BaseModel):
    imagem_base64: str
    tipo_documento: Optional[str] = "auto"

P_CLASS = """Especialista em triagem juridica brasileira.
REGRAS: rmc_rcc codigo 322/217 extrato INSS=viabilidade minimo 85.
rmc_rcc desconto nao reconhecido INSS=viabilidade minimo 75.
trabalhista CTPS+holerites=viabilidade minimo 70.
sabesp contas cobranca=viabilidade minimo 80.
ir_doenca_grave laudo CID=viabilidade minimo 85.
Sem doc critico: reduza ate 20pts, minimo 50. Prescrito=0 recusar.
Teses: rmc_rcc trabalhista auxilio_acidente plano_saude ir_doenca_grave iptu itbi sabesp_fator_k
JSON: {"tese":"","tese_label":"","viabilidade":0,"viabilidade_justificativa":"","documentos_recebidos":[],"documentos_faltantes":[],"documentos_criticos_faltantes":false,"prescrito":false,"prescricao_observacao":null,"valor_estimado_min":0,"valor_estimado_max":0,"ticket_honorarios_estimado":0,"urgencia":"alta","recomendacao":"aceitar","notas_advogado":"rmc_rcc: Tema 1328 STJ sobrestado dano moral so subsidiario","mensagem_cliente":""}
rmc_rcc sem valor: min=3000 max=12000 ticket=2250"""

P_CONT = """Advogado especialista contratos honorarios Brasil.
Contrato honorarios de exito 30% valor liquido. Art.22 Lei 8906/94. Lei 14063/2020.
Estrutura: 1)Qualificacao 2)Objeto 3)Obrigacoes escritorio 4)Obrigacoes cliente
5)Honorarios 30% exito sem antecipacao 6)Prazo rescisao 7)Procuracao ad judicia et extra 8)Foro 9)Assinaturas"""

P_PET = """Advogado advocacia de massa brasileira.
Peticoes iniciais COMPLETAS: I)Enderecamento II)Qualificacao III)Fatos IV)Direito V)Pedidos VI)Valor VII)Provas VIII)Requerimentos
rmc_rcc: arts 6III 42 CDC Sumula 297 STJ IN INSS 28/2008. Tema 929 STJ dobro pos 30/03/2021. Tema 1328 SOBRESTADO dano moral SO subsidiario. Tutela urgencia suspensao desconto.
trabalhista: CLT OJs TST horas extras 50% reflexos TRCT FGTS multa 40%.
sabesp: Sumula 193 TJSP retroativo 5 anos IPCA."""

P_JULIA = """Julia assistente juridica Addebitare. Portugues coloquial. Max 3 paragrafos. Max 2 emojis.
ESCALAR: [motivo] se mencionar cancelar desistir intimacao citacao audiencia recurso OAB advogado.
Nunca garanta resultado ou prazo."""

P_EXT = """Especialista documentos juridicos brasileiros.
JSON: {"tipo_documento":"extrato_inss|ctps|holerite|contrato_rmc_rcc|conta_sabesp|laudo_medico","confianca_leitura":0,"dados":{},"alertas":[],"tese_sugerida":""}"""

TESES = {"rmc_rcc":"nulidade contrato RMC/RCC repeticao indebito dobro","trabalhista":"reclamatoria trabalhista verbas rescisorias","auxilio_acidente":"concessao restituicao B36 INSS","plano_saude":"negativa cobertura plano saude","ir_doenca_grave":"isencao IRPF doenca grave Lei 7713/88","iptu":"revisao IPTU restituicao","itbi":"restituicao ITBI excesso","sabesp_fator_k":"nulidade Fator K SABESP Sumula 193 TJSP"}
JURIS = {"rmc_rcc":"[STJ Tema 929] EAREsp 676.608/RS repeticao dobro pos 30/03/2021\n[STJ Tema 1328] REsp 2.145.244/SC SOBRESTADO dano moral subsidiario\n[TJSP] AC 1016333-05.2020.8.26.0068 Des Carlos Abrao conversao RMC\n[STJ] Sumula 297 CDC instituicoes financeiras","trabalhista":"[TST] Sumulas 338 85 OJ 394","sabesp_fator_k":"[TJSP] Sumula 193 Fator K indevido\n[TJSP] prescricao 5 anos art 27 CDC","ir_doenca_grave":"[STJ] REsp 1.116.620/BA isencao IR doenca grave\n[STJ] REsp 1.507.970 prazo 5 anos CTN","auxilio_acidente":"[STJ] REsp 1.294.074/SP auxilio B36\n[STJ] Sumula 491","plano_saude":"[STJ] Sumulas 608 [STJ] REsp 1.733.013/PR dano moral in re ipsa","iptu":"[STF] RE 591.795 planta generica lei","itbi":"[STF] RE 1.294.969 Tema 1.113 base calculo ITBI valor declarado"}

def _claude(s,u,m="claude-sonnet-4-20250514",t=4000):
    if not claude: raise HTTPException(500,"ANTHROPIC_API_KEY nao configurada")
    return claude.messages.create(model=m,max_tokens=t,system=s,messages=[{"role":"user","content":u}]).content[0].text

def _juris(tese,fatos):
    if not (PINECONE_KEY and OPENAI_KEY): return JURIS.get(tese,"[STJ] Sumula 297")
    try:
        from openai import OpenAI; from pinecone import Pinecone
        oa=OpenAI(api_key=OPENAI_KEY); pc=Pinecone(api_key=PINECONE_KEY); idx=pc.Index(PINECONE_INDEX)
        emb=oa.embeddings.create(model="text-embedding-3-large",input=fatos[:6000],dimensions=3072).data[0].embedding
        res=idx.query(vector=emb,top_k=5,filter={"tese_interna":{"$eq":tese}},include_metadata=True)
        ls=[f"[{m[chr(109)+chr(101)+chr(116)+chr(97)+chr(100)+chr(97)+chr(116)+chr(97)].get(chr(116)+chr(114)+chr(105)+chr(98)+chr(117)+chr(110)+chr(97)+chr(108))}] {m[chr(109)+chr(101)+chr(116)+chr(97)+chr(100)+chr(97)+chr(116)+chr(97)].get(chr(110)+chr(117)+chr(109)+chr(101)+chr(114)+chr(111)+chr(80)+chr(114)+chr(111)+chr(99)+chr(101)+chr(115)+chr(115)+chr(111))}\n{m[chr(109)+chr(101)+chr(116)+chr(97)+chr(100)+chr(97)+chr(116)+chr(97)].get(chr(101)+chr(109)+chr(101)+chr(110)+chr(116)+chr(97),chr(32))[:200]}" for m in res["matches"] if m["score"]>=0.72]
        return "\n".join(ls) if ls else JURIS.get(tese,"")
    except Exception as e: log.warning(f"Pinecone fallback: {e}"); return JURIS.get(tese,"")

def _cs(txt,nome,email,cpf):
    if not CLICKSIGN_KEY: return {"status":"simulado","link":f"https://app.clicksign.com/sign/DEMO-{cpf[-4:]}","doc_key":"DEMO"}
    h={"Content-Type":"application/json"}; b="https://app.clicksign.com/api/v1"
    d64=base64.b64encode(txt.encode()).decode()
    r1=requests.post(f"{b}/documents?access_token={CLICKSIGN_KEY}",headers=h,json={"document":{"path":f"/c/{cpf.replace(chr(46),chr(32)).replace(chr(45),chr(32)).replace(chr(32),chr(95))}_{datetime.now().strftime(chr(37)+chr(89)+chr(37)+chr(109)+chr(37)+chr(100)+chr(37)+chr(72)+chr(37)+chr(77)+chr(37)+chr(83))}.txt","content_base64":f"data:text/plain;base64,{d64}","auto_close":True,"locale":"pt-BR"}})
    r1.raise_for_status(); dk=r1.json()["document"]["key"]
    r2=requests.post(f"{b}/signers?access_token={CLICKSIGN_KEY}",headers=h,json={"signer":{"email":email,"name":nome,"documentation":cpf,"has_documentation":True}})
    r2.raise_for_status(); sk=r2.json()["signer"]["key"]
    r3=requests.post(f"{b}/lists?access_token={CLICKSIGN_KEY}",headers=h,json={"list":{"document_key":dk,"signer_key":sk,"sign_as":"party"}})
    r3.raise_for_status(); return {"status":"enviado","link":r3.json()["list"]["url"],"doc_key":dk}

@app.get("/health")
def health(): return {"status":"ok","version":"2.0.0","timestamp":datetime.now().isoformat(),"servicos":{"claude":"ativo" if ANTHROPIC_KEY else "sem chave","openai":"ativo" if OPENAI_KEY else "nao configurado","pinecone":"ativo" if PINECONE_KEY else "fallback","clicksign":"ativo" if CLICKSIGN_KEY else "simulado","datajud":"ativo" if DATAJUD_KEY else "nao configurado"}}

@app.post("/classificar")
async def classificar(b: LeadInput):
    u=f"DESCRICAO:\n{b.descricao}"
    if b.documentos_ocr: u+=f"\nDOCS:\n{b.documentos_ocr}"
    if b.nome: u+=f"\nCliente: {b.nome}"
    raw=_claude(P_CLASS,u,m="claude-haiku-4-5-20251001",t=1200)
    try:
        r=json.loads(raw.replace("```json","").replace("```","").strip())
        log.info(f"Triagem ok: tese={r.get('tese')} viabilidade={r.get('viabilidade')}%")
        return r
    except: raise HTTPException(500,f"JSON error: {raw[:200]}")

@app.post("/contrato")
async def contrato(b: ContratoInput):
    u=f"Gere contrato:\nESCRITORIO: Addebitare Capital Advocacia\nCLIENTE: {b.nome} CPF:{b.cpf} Email:{b.email} Tel:{b.telefone}\nOBJETO: {TESES.get(b.tese,b.tese)}\nVALOR: R${b.valor_min:,.0f} a R${b.valor_max:,.0f}\nHONORARIOS: 30% exito puro\nDATA: {datetime.now().strftime(chr(37)+chr(100)+chr(47)+chr(37)+chr(109)+chr(47)+chr(37)+chr(89))}"
    txt=_claude(P_CONT,u,t=4000)
    cs=_cs(txt,b.nome,b.email,b.cpf)
    return {"contrato_texto":txt,"clicksign_link":cs["link"],"clicksign_status":cs["status"],"clicksign_doc_key":cs["doc_key"],"mensagem_wpp":f"Ola {b.nome.split()[0]}! Seu contrato esta pronto.\n\n{cs[chr(108)+chr(105)+chr(110)+chr(107)]}\n\nAssim que assinar, comecamos a trabalhar!"}

@app.post("/peticao")
async def peticao(b: PeticaoInput):
    vt=b.valor_principal+b.dano_moral
    j="JEC" if vt<=56480 else "Vara Civel"
    p={"auxilio_acidente":"SEEU","trabalhista":"PJe TRT"}.get(b.tese,"eSAJ/PJe TJSP")
    jr=_juris(b.tese,b.fatos)
    u=f"Peticao:\nREQ: {b.nome} CPF:{b.cpf}\nREU: {b.reu}\nTESE: {b.tese} - {TESES.get(b.tese,b.tese)}\nJUIZO: {j} PROT: {p}\nVALOR: R${vt:,.2f}\nFATOS:\n{b.fatos}\nJURIS:\n{jr}"
    txt=_claude(P_PET,u,t=7000)
    return {"peticao_texto":txt,"juizo":j,"protocolo":p,"valor_causa":vt,"rag_ativo":bool(PINECONE_KEY and OPENAI_KEY)}

@app.post("/wpp")
async def wpp(b: WppInput):
    sys=f"{P_JULIA}\nCLIENTE: {b.nome_cliente or chr(67)+chr(108)+chr(105)+chr(101)+chr(110)+chr(116)+chr(101)}\nPROC: {b.numero_processo or chr(101)+chr(109)+chr(32)+chr(116)+chr(114)+chr(97)+chr(109)+chr(105)+chr(116)+chr(97)+chr(99)+chr(97)+chr(111)}\nFASE: {b.fase_processo}"
    msgs=(b.historico or [])[-10:]
    msgs.append({"role":"user","content":b.mensagem})
    resp=claude.messages.create(model="claude-haiku-4-5-20251001",max_tokens=450,system=sys,messages=msgs).content[0].text
    esc=resp.upper().startswith("ESCALAR:") or any(g in b.mensagem.lower() for g in ["cancelar","desistir","intimacao","citacao","audiencia","recurso","oab","advogado"])
    mot=None
    if resp.upper().startswith("ESCALAR:"): mot=resp.split(":",1)[1].strip().split("\n")[0]; resp="Entendi! Vou transferir voce para um advogado agora!"
    return {"resposta":resp,"escalar":esc,"motivo_escalonamento":mot}

@app.post("/extrair")
async def extrair(b: ExtrairInput):
    if not OPENAI_KEY: raise HTTPException(400,"OPENAI_API_KEY necessaria para OCR")
    try:
        from openai import OpenAI; oa=OpenAI(api_key=OPENAI_KEY)
        r=oa.chat.completions.create(model="gpt-4o",messages=[{"role":"system","content":P_EXT},{"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b.imagem_base64}","detail":"high"}},{"type":"text","text":"Extraia os dados."}]}],max_tokens=2000,temperature=0.1,response_format={"type":"json_object"})
        return json.loads(r.choices[0].message.content)
    except Exception as e: raise HTTPException(500,str(e))

@app.post("/atualizar-rag")
async def rag(dias: int = 90):
    if not all([OPENAI_KEY,PINECONE_KEY,DATAJUD_KEY]): return {"status":"pulado","motivo":"Configure OPENAI PINECONE DATAJUD keys"}
    try:
        from openai import OpenAI; from pinecone import Pinecone, ServerlessSpec; from datetime import timedelta
        oa=OpenAI(api_key=OPENAI_KEY); pc=Pinecone(api_key=PINECONE_KEY)
        if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]: pc.create_index(name=PINECONE_INDEX,dimension=3072,metric="cosine",spec=ServerlessSpec(cloud="aws",region="us-east-1"))
        idx=pc.Index(PINECONE_INDEX)
        di=(datetime.now()-timedelta(days=dias)).strftime("%Y-%m-%d"); df=datetime.now().strftime("%Y-%m-%d")
        qs={"rmc_rcc":"reserva margem consignavel RMC INSS nulidade","trabalhista":"horas extras FGTS verbas rescisorias","sabesp_fator_k":"SABESP Fator K Sumula 193","ir_doenca_grave":"isencao IR doenca grave Lei 7713","auxilio_acidente":"auxilio acidente B36 INSS"}
        total=0
        for tese,q in qs.items():
            r=requests.post("https://api-publica.datajud.cnj.jus.br/api_publica_tjsp/_search",json={"size":20,"query":{"bool":{"must":[{"multi_match":{"query":q,"fields":["ementa"]}}],"filter":[{"range":{"dataJulgamento":{"gte":di,"lte":df}}}]}},"sort":[{"dataJulgamento":{"order":"desc"}}]},headers={"Authorization":f"ApiKey {DATAJUD_KEY}"},timeout=30)
            if r.status_code!=200: continue
            vs=[]
            for h in r.json().get("hits",{}).get("hits",[]):
                e=h["_source"].get("ementa","")
                if len(e)<50: continue
                emb=oa.embeddings.create(model="text-embedding-3-large",input=e[:6000],dimensions=3072).data[0].embedding
                vs.append({"id":f"tjsp_{h[chr(95)+chr(105)+chr(100)]}_{tese}","values":emb,"metadata":{"tribunal":"TJSP","tese_interna":tese,"ementa":e[:800],"numero_processo":h["_source"].get("numeroProcesso",""),"data_julgamento":str(h["_source"].get("dataJulgamento",""))[:10],"resultado":"favoravel" if any(w in e.lower() for w in ["provido","procedente"]) else "desfavoravel"}})
            for i in range(0,len(vs),100): idx.upsert(vectors=vs[i:i+100])
            total+=len(vs)
        return {"status":"ok","acordaos_indexados":total,"periodo":f"{di} a {df}"}
    except Exception as e: raise HTTPException(500,str(e))