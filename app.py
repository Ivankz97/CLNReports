import os
import json 
import re 
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from urllib.parse import quote_plus 
import streamlit as st

# --- 1. Initialize Components

PERSIST_DIRECTORY = "./culiacan_incidentes_db"

@st.cache_resource
def initialize_system():
    """Initialize LLM, Embeddings and Chroma"""
    
    try:
        api_key = st.secrets["google_api_key"]
    except KeyError:
        raise KeyError("The 'google_api_key' is not configured in the Secrets Streamlit")

    try:
        # Send keys to LangChain Constructors
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key) 

    except Exception as e:
        raise ConnectionError(f"Connection failed. Error: {e}")

    # B. Vector Store (Load or creation of ChromaDB)
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    else:
        initial_incidents = [
            "Incidente: Semáforo dañado. Ubicación: Av. Obregón, Culiacán. Fecha: 2025-11-01",
            "Incidente: Hoyo en el pavimento. Ubicación: Malecón Nuevo, Culiacán. Se requiere bacheo. Fecha: 2025-10-28",
            "Incidente: Reporte de choque en alturas del sur. Ubicación: Alturas del Sur, Culiacán. Fecha: 2025-11-04",
            "Incidente: Reporte de robo en alturas del sur. Ubicación: Alturas del Sur, Culiacán. Fecha: 2025-11-04",
            "Incidente: Asalto en barrancos. Ubicación: Barrancos, Culiacán. Fecha: 2025-11-04"
        ]
        initial_documents = [Document(page_content=t) for t in initial_incidents]
        
        vectorstore = Chroma.from_documents(
            initial_documents, 
            embedding=embeddings, 
            persist_directory=PERSIST_DIRECTORY
        )
        vectorstore.persist() 
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    
    return llm, vectorstore, retriever


try:
    llm, vectorstore, retriever = initialize_system()
    st.sidebar.info("✅ Sistema y ChromaDB listos.")
except (KeyError, ConnectionError) as e:
    st.error(f"❌ ERROR CRÍTICO: {e}")
    st.stop()


# C. LLM ROL FOR LOCATION EXTRACTION
LOCATION_EXTRACTOR_ROLE = SystemMessage(
    content="Eres un extractor de texto. Tu única tarea es analizar el mensaje del usuario e identificar la ubicación específica. Responde ESTRICTAMENTE con un objeto JSON válido, **encerrando la respuesta SÓLO dentro de triple backtick y la palabra 'json'** (ejemplo: ```json{\"ubicacion\": \"valor\"}```). La ubicación debe ser el nombre del lugar, seguido de ', Culiacán' para mayor precisión. Si no detectas una ubicación válida, responde estrictamente con el JSON: ```json{\"ubicacion\": \"NO_UBICACION\"}```"
)

# --- 2. Utility functions ---

def get_osm_search_link(location_name):
    """Generate the URL for OpenStreetMap."""
    search_query = quote_plus(location_name)
    return f"https://www.openstreetmap.org/search?query={search_query}"

def extract_location_from_report(report_text, default_location):
    """Extract saved location from reports"""
    match = re.search(r"Ubicación:\s*(.+?)\.\s*Fecha:", report_text)
    return match.group(1).strip() if match else default_location


# --- 3. Main logic for Streamlit ---

st.title("🚨 Sistema de Gestión de Incidentes de Culiacán")

st.sidebar.header("Opciones")
choice = st.sidebar.radio("Elige una acción:", ('📝 Reportar Incidente', '🔍 Consultar Incidentes'))

user_input = st.text_area(
    "Ingresa la descripción y ubicación del incidente (ej. Choque en Av. Obregón)", 
    key="main_input",
    height=100
)

if st.button("Procesar Solicitud", type="primary"):
    
    if not user_input:
        st.warning("Por favor, ingresa una descripción para procesar.")
        st.stop()

    # 1. Get Location
    try:
        location_messages = [LOCATION_EXTRACTOR_ROLE, HumanMessage(content=user_input)]
        
        with st.spinner('🌎 Buscando ubicación con Gemini...'):
            location_json_str = llm.invoke(location_messages).content

        # CLEAN AND JSON PARSER
        cleaned_json_str = location_json_str.strip()
        if cleaned_json_str.startswith("```json"):
            cleaned_json_str = cleaned_json_str[7:]
        if cleaned_json_str.endswith("```"):
            cleaned_json_str = cleaned_json_str[:-3]
        
        try:
            location_data = json.loads(cleaned_json_str.strip())
        except json.JSONDecodeError:
            location_data = {'ubicacion': 'NO_UBICACION'}
            
        if location_data['ubicacion'] == 'NO_UBICACION':
            st.error("🚨 ERROR: No se pudo detectar una ubicación válida en Culiacán. Intenta ser más específico.")
            st.stop()
        
        ubicacion_detectada = location_data['ubicacion']
        st.info(f"🔎 Ubicación detectada: **{ubicacion_detectada}**")

        
        # --- OPTION 1: REPORT ---
        if choice == '📝 Reportar Incidente':
            incident_report = f"Incidente: {user_input}. Ubicación: {ubicacion_detectada}. Fecha: {os.environ.get('CURRENT_DATE', '2025-11-05')}"
            
            with st.spinner('💾 Guardando reporte en ChromaDB...'):
                vectorstore.add_texts([incident_report])
                vectorstore.persist() 
            st.success(f"✅ ¡Incidente reportado con éxito! **{ubicacion_detectada}** ha sido añadido a la memoria.")

        # --- OPTION 2: Search---
        elif choice == '🔍 Consultar Incidentes':
            st.subheader(f"Resultados de la consulta cerca de '{ubicacion_detectada}'")
            
            with st.spinner('🔍 Buscando reportes similares en la memoria...'):
                retrieved_documents = retriever.invoke(user_input)
            
            if retrieved_documents:
                
                main_report = retrieved_documents[0]
                location_for_map = extract_location_from_report(main_report.page_content, ubicacion_detectada)
                osm_link = get_osm_search_link(location_for_map)
                
                # Results
                st.markdown("---")
                st.markdown(f"**🥇 Reporte más relevante encontrado:**")
                st.code(main_report.page_content, language='text')

                st.subheader("🗺️ Visualización de Mapa (OpenStreetMap)")
                st.markdown(f"Busca la ubicación más relevante: **`{location_for_map}`**")
                st.link_button("Abrir Mapa en Navegador", osm_link)

                if len(retrieved_documents) > 1:
                     st.markdown("---")
                     st.markdown(f"**Otros {len(retrieved_documents) - 1} reportes relacionados:**")
                     for i, doc in enumerate(retrieved_documents[1:]):
                        st.markdown(f"- **Reporte #{i+2}:** `{doc.page_content[:90]}...`")
            else:
                st.warning("No se encontraron incidentes cercanos a esa ubicación en la memoria.")
            

    except Exception as e:
        st.error(f"🛑 Se produjo un error: {e}")
        st.stop()

# Save to Chroma if something has changed
if 'vectorstore' in locals():
    try:
        vectorstore.persist()
    except Exception:
        pass