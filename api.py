from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Audio Noise Effects API",
    description="API pour le traitement audio et l'ajout d'effets sonores",
    version="1.0.0"
)


@app.get("/")
async def hello_world():
    """
    Endpoint de test simple qui retourne un message Hello World
    """
    return {"message": "Hello World"}


@app.get("/health")
async def health_check():
    """
    Endpoint de vérification de l'état de l'API
    """
    return {"status": "healthy"}


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    """
    Endpoint pour uploader une vidéo .mp4
    
    Args:
        video: Fichier vidéo au format .mp4
        
    Returns:
        JSON indiquant si la vidéo a été reçue ou non
    """
    try:
        # Vérifier si un fichier a été envoyé
        if not video:
            return JSONResponse(
                status_code=400,
                content={"status": "video non received", "error": "Aucun fichier fourni"}
            )
        
        # Vérifier l'extension du fichier
        if not video.filename.endswith('.mp4'):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "video non received", 
                    "error": "Le fichier doit être au format .mp4",
                    "filename": video.filename
                }
            )
        
        # Vérifier le type MIME
        if video.content_type not in ["video/mp4", "application/octet-stream"]:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "video non received",
                    "error": f"Type de fichier non valide: {video.content_type}",
                    "filename": video.filename
                }
            )
        
        # Lire le contenu pour vérifier que le fichier n'est pas vide
        contents = await video.read()
        if len(contents) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "video non received",
                    "error": "Le fichier est vide",
                    "filename": video.filename
                }
            )
        
        # Si tout est OK, retourner succès
        return JSONResponse(
            status_code=200,
            content={
                "status": "video received",
                "filename": video.filename,
                "size_bytes": len(contents),
                "content_type": video.content_type
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "video non received",
                "error": f"Erreur lors du traitement: {str(e)}"
            }
        )

