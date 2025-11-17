import os
import json 
import re 
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from urllib.parse import quote_plus 
import streamlit as st
from datetime import datetime # Imported for date sorting

# --- 1. Component Initialization (Caching and Security) ---

PERSIST_DIRECTORY = "./culiacan_incidentes_db"

# We pass the cache function itself to use its clear() method later
@st.cache_resource
def initialize_system():
  """Initializes LLM, Embeddings, and Chroma DB (Load or Create)."""
  # Securely read the API key from Streamlit secrets
  try:
    api_key = st.secrets["google_api_key"]
  except KeyError:
    raise KeyError("The 'google_api_key' is not configured in Streamlit secrets. Check your configuration!")

  try:
    # Pass the key to LangChain constructors
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key) 

  except Exception as e:
    raise ConnectionError(f"Google connection failed. Error: {e}")

  # B. Vector Store (Load or Create ChromaDB)
  if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
    # Load existing database
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
  else:
    # Create initial sample data
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
  
  # Configure retriever for similarity search
  retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
  
  return llm, vectorstore, retriever

try:
    # We call it without the decorator for the first time
    llm, vectorstore, retriever = initialize_system()
except (KeyError, ConnectionError) as e:
    st.error(f"❌ ERROR CRÍTICO: {e}")
    st.stop()


# C. LLM ROLE FOR LOCATION EXTRACTION
LOCATION_EXTRACTOR_ROLE = SystemMessage(
    content="Eres un extractor de texto. Tu única tarea es analizar el mensaje del usuario e identificar la ubicación específica. Responde ESTRICTAMENTE con un objeto JSON válido, **encerrando la respuesta SÓLO dentro de triple backtick y la palabra 'json'** (ejemplo: ```json{\"ubicacion\": \"valor\"}```). La ubicación debe ser el nombre del lugar, seguido de ', Culiacán' para mayor precisión. Si no detectas una ubicación válida, responde estrictamente con el JSON: ```json{\"ubicacion\": \"NO_UBICACION\"}```"
)

# --- 2. Utility Functions ---

def get_osm_search_link(location_name):
    """Generates the URL for OpenStreetMap search."""
    search_query = quote_plus(location_name)
    return f"https://www.openstreetmap.org/search?query={search_query}"

def extract_location_from_report(report_text):
    """Extracts the saved location from the report using regex."""
    match = re.search(r"Ubicación:\s*(.+?)\.\s*Fecha:", report_text)
    return match.group(1).strip() if match else "Ubicación Desconocida"

def extract_date_from_report(report_text):
    """Extracts the date (YYYY-MM-DD) from the report content."""
    match = re.search(r"Fecha:\s*(\d{4}-\d{2}-\d{2})", report_text)
    return match.group(1).strip() if match else None

def get_recent_reports(vectorstore, top_n=10):
    """Retrieves all reports, sorts them by date, and returns the top N most recent."""
    
    # Access the underlying Chroma collection directly to retrieve all documents.
    try:
        results = vectorstore._collection.get(
            include=['documents'] 
        )
    except AttributeError:
        st.error("Error accessing the internal Chroma collection.")
        return []

    reports = []
    for doc_id, doc_content in zip(results['ids'], results['documents']):
        date_str = extract_date_from_report(doc_content)
        if date_str:
            try:
                # Convert date string to datetime object for sorting
                reports.append({
                    'content': doc_content,
                    'date': datetime.strptime(date_str, '%Y-%m-%d')
                })
            except ValueError:
                continue # Skip documents with incorrect date format
    
    # Sort by date descending (most recent first)
    reports.sort(key=lambda x: x['date'], reverse=True)
    
    return reports[:top_n]


# --- 3. Main Streamlit Logic ---

st.title("🚨 Sistema de Gestión de Incidentes de Culiacán")

st.sidebar.header("Opciones")
# Radio button for choosing action
choice = st.sidebar.radio("Elige una acción:", ('📝 Reportar Incidente', '🔍 Consultar Incidentes', '📅 Ver Reportes Recientes'))

# --- Input Block (Applies only to Reporting/Consulting) ---
if choice in ('📝 Reportar Incidente', '🔍 Consultar Incidentes'):
    user_input = st.text_area(
        "Ingresa la descripción y ubicación del incidente (ej. Choque en Av. Obregón)", 
        key="main_input",
        height=100
    )
    
    if st.button("Procesar Solicitud", type="primary"):
        
        if not user_input:
            st.warning("Por favor, ingresa una descripción para procesar.")
            st.stop()

        # 1. Location Extraction
        try:
            location_messages = [LOCATION_EXTRACTOR_ROLE, HumanMessage(content=user_input)]
            
            with st.spinner('🌎 Buscando ubicación...'):
                location_json_str = llm.invoke(location_messages).content

            # JSON cleaning and parsing logic
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

            
            # --- Logic for Option 1: REPORT ---
            if choice == '📝 Reportar Incidente':
                # Use current date for the report
                current_date = datetime.now().strftime('%Y-%m-%d')
                incident_report = f"Incidente: {user_input}. Ubicación: {ubicacion_detectada}. Fecha: {current_date}"
                
                with st.spinner('💾 Guardando reporte...'):
                    vectorstore.add_texts([incident_report])
                    vectorstore.persist() 
                    
                # a initialize_system
                initialize_system.clear()
                
                st.success(f"✅ ¡Incidente reportado con éxito! Fecha: **{current_date}**")

            # --- Logic for Option 2: CONSULT ---
            elif choice == '🔍 Consultar Incidentes':
                st.subheader(f"Resultados de la consulta cerca de '{ubicacion_detectada}'")
                
                with st.spinner('🔍 Buscando reportes similares en la memoria...'):
                    # Note: Retriever uses the current (possibly cached) vectorstore instance
                    retrieved_documents = retriever.invoke(user_input)
                
                if retrieved_documents:
                    main_report = retrieved_documents[0]
                    location_for_map = extract_location_from_report(main_report.page_content)
                    osm_link = get_osm_search_link(location_for_map)
                    
                    # Displaying the most relevant result
                    st.markdown("---")
                    st.markdown(f"**🥇 Reporte más relevante encontrado:**")
                    st.code(main_report.page_content, language='text')

                    st.subheader("🗺️ Visualización de Mapa (OpenStreetMap)")
                    st.markdown(f"Busca la ubicación más relevante: **`{location_for_map}`**")
                    st.link_button("Abrir Mapa en Navegador", osm_link)

                    # Displaying other related results
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

# --- Logic for Option 3: RECENT REPORTS ---
elif choice == '📅 Ver Reportes Recientes':
    st.header("📅 Reportes Más Recientes (Top 10)")
    
    with st.spinner('⏳ Recuperando y ordenando todos los reportes...'):
        # To ensure the most recent data is displayed, we must call initialize_system 
        # again if the cache was cleared (which happens on next script run after reporting).
        # We rely on Streamlit's rerun mechanism to update the data here.
        recent_reports = get_recent_reports(vectorstore, top_n=10)
    
    if recent_reports:
        for i, report in enumerate(recent_reports):
            date_str = report['date'].strftime('%Y-%m-%d')
            
            # Clean content for display (remove redundant date text)
            content_display = report['content'].replace(f' Fecha: {date_str}', '')
            
            st.markdown(f"**{i+1}.** **Fecha:** **`{date_str}`** | **Contenido:** {content_display}")
            st.markdown("---")
    else:
        st.info("No se encontraron reportes en la base de datos.")


# Ensures Chroma persists data upon completion/exit of the Streamlit session
if 'vectorstore' in locals():
    try:
        vectorstore.persist()
    except Exception:
        pass