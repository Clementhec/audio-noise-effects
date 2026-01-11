import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Récupérer les données du formulaire
    const formData = await request.formData();
    
    // Faire l'appel à votre endpoint local
    const response = await fetch('http://localhost:8000/upload-video', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Erreur du serveur: ${response.status}`);
    }

    // Récupérer la réponse
    const data = await response.json();

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Erreur lors de l\'appel à l\'API:', error);
    return NextResponse.json(
      { error: 'Erreur lors du traitement de la requête' },
      { status: 500 }
    );
  }
}


