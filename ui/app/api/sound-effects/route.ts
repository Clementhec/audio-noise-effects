import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Récupérer les données du formulaire
    const formData = await request.formData();
    
    // Renommer 'prompt' en 'user_prompt' pour correspondre à l'API FastAPI
    const prompt = formData.get('prompt');
    if (prompt) {
      formData.delete('prompt');
      formData.append('user_prompt', prompt);
    }
    
    // Faire l'appel au serveur FastAPI
    const response = await fetch('http://localhost:8000/process-video', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Erreur du serveur: ${response.status}`);
    }

    // La réponse est maintenant une vidéo
    const videoBlob = await response.blob();
    
    // Retourner un succès avec un message
    return NextResponse.json({ 
      success: true,
      message: 'Vidéo traitée avec succès',
      videoProcessed: true 
    }, { status: 200 });
    
  } catch (error) {
    console.error('Erreur lors de l\'appel à l\'API:', error);
    return NextResponse.json(
      { error: 'Erreur lors du traitement de la requête' },
      { status: 500 }
    );
  }
}


