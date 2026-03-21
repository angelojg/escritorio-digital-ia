"""
Escritorio Digital IA - Backend Principal v2
Correcoes: prompt classificador, endpoint /extrair, logica de viabilidade
"""
import os, json, logging, base64
from datetime import datetime
from typing import Optional
import anthropic, requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel