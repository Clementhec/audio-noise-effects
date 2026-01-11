"use client"

import type React from "react"
import Image from "next/image"
import { useState } from "react"
import { Upload, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import VideoLoading from "@/components/video-loading"

interface VideoUploadProps {
  onComplete: (videoFile: File, videoUrl: string) => void
}

export default function VideoUpload({ onComplete }: VideoUploadProps) {
  const [state, setState] = useState<"upload" | "processing">("upload")
  const [prompt, setPrompt] = useState("")
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [progress, setProgress] = useState(0)

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setVideoFile(file)
      
      // Appel à l'API upload_file
      try {
        const formData = new FormData()
        formData.append('video', file)
        
        console.log('Upload du fichier:', file.name)
        
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        })
        
        if (!response.ok) {
          const errorData = await response.json()
          console.error('Erreur du serveur:', errorData)
          alert(`Erreur: ${errorData.error || 'Erreur lors de l\'upload'}\n\nAssurez-vous que votre serveur backend est démarré sur http://localhost:8000`)
          return
        }
        
        const data = await response.json()
        console.log('Réponse de l\'API:', data)
      } catch (error) {
        console.error('Erreur lors de l\'upload:', error)
        alert('Erreur de connexion. Vérifiez que votre serveur backend est démarré sur http://localhost:8000')
      }
    }
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith("video/")) {
      setVideoFile(file)
      
      // Appel à l'API upload-video
      try {
        const formData = new FormData()
        formData.append('video', file)
        
        console.log('Upload du fichier:', file.name)
        
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        })
        
        if (!response.ok) {
          const errorData = await response.json()
          console.error('Erreur du serveur:', errorData)
          alert(`Erreur: ${errorData.error || 'Erreur lors de l\'upload'}\n\nAssurez-vous que votre serveur backend est démarré sur http://localhost:8000`)
          return
        }
        
        const data = await response.json()
        console.log('Réponse de l\'API:', data)
      } catch (error) {
        console.error('Erreur lors de l\'upload:', error)
        alert('Erreur de connexion. Vérifiez que votre serveur backend est démarré sur http://localhost:8000')
      }
    }
  }

  const handleGenerate = () => {
    if (videoFile && prompt) {
      setState("processing")
      // Simulate processing
      let progressValue = 0
      const interval = setInterval(() => {
        progressValue += 5
        setProgress(progressValue)
        if (progressValue >= 100) {
          clearInterval(interval)
          const url = URL.createObjectURL(videoFile)
          onComplete(videoFile, url)
        }
      }, 100)
    }
  }

  if (state === "processing") {
    return <VideoLoading progress={progress} />
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-8 bg-gradient-to-br from-[#8B7BC7] via-[#A594D8] to-[#9B8BD3] relative overflow-hidden">
      {/* Blob pattern background */}
      <div className="absolute inset-0 opacity-40 pointer-events-none">
        <div className="absolute top-20 left-20 w-48 h-48 bg-[#7B6BB7] rounded-full blur-3xl animate-pulse" />
        <div className="absolute top-60 right-32 w-64 h-64 bg-[#9B8BD3] rounded-full blur-3xl animate-pulse delay-300" />
        <div className="absolute bottom-40 left-1/3 w-56 h-56 bg-[#8B7BC7] rounded-full blur-3xl animate-pulse delay-500" />
        <div className="absolute bottom-20 right-20 w-40 h-40 bg-[#A594D8] rounded-full blur-3xl animate-pulse delay-700" />
        <div className="absolute top-1/3 left-10 w-32 h-32 bg-[#DFFF00] rounded-full blur-2xl opacity-20 animate-pulse delay-1000" />
        <div className="absolute bottom-1/3 right-10 w-36 h-36 bg-[#FF69B4] rounded-full blur-2xl opacity-20 animate-pulse delay-700" />
      </div>

      <div className="w-full max-w-3xl space-y-8 relative z-10">
        {/* Header with Logo */}
        <div className="text-center space-y-6">
          <div className="flex justify-center">
            <Image 
              src="/soundeasy.gif" 
              alt="SoundEasy" 
              width={350} 
              height={175}
              className="drop-shadow-2xl"
              unoptimized
            />
          </div>
          <p className="text-xl text-white/90 font-medium drop-shadow-md" style={{ fontFamily: 'Comic Sans MS, cursive, sans-serif' }}>
            {"Upload your video and describe the audio effects you want to add ✨"}
          </p>
        </div>

        {/* Upload Area */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className={cn(
            "border-4 border-dashed rounded-3xl p-12 transition-all duration-300 backdrop-blur-sm",
            videoFile 
              ? "border-[#DFFF00] bg-[#DFFF00]/20 shadow-[0_0_30px_rgba(223,255,0,0.3)]" 
              : "border-white/50 bg-white/10 hover:border-[#FF69B4] hover:bg-[#FF69B4]/10 hover:shadow-[0_0_20px_rgba(255,105,180,0.3)]",
          )}
        >
          <div className="flex flex-col items-center gap-4 text-center">
            <div className="rounded-full bg-[#DFFF00] p-5 shadow-lg">
              <Upload className="h-10 w-10 text-[#8B7BC7]" />
            </div>
            <div className="space-y-2">
              <p className="text-xl font-bold text-white drop-shadow-md" style={{ fontFamily: 'Comic Sans MS, cursive, sans-serif' }}>
                {videoFile ? `🎬 ${videoFile.name}` : "Drop your video here!"}
              </p>
              <p className="text-sm text-white/80 font-medium">
                {"or click to browse • MP4, MOV, AVI • Max 500MB"}
              </p>
            </div>
            <input type="file" accept="video/*" onChange={handleFileUpload} className="hidden" id="video-upload" />
            <label htmlFor="video-upload">
              <Button 
                variant="secondary" 
                size="lg" 
                asChild
                className="bg-white hover:bg-[#DFFF00] text-[#8B7BC7] font-bold text-lg px-8 py-6 rounded-full shadow-lg transition-all duration-300 hover:scale-105 hover:shadow-xl"
              >
                <span style={{ fontFamily: 'Comic Sans MS, cursive, sans-serif' }}>{"Choose a file"}</span>
              </Button>
            </label>
          </div>
        </div>

        {/* Prompt Area */}
        <div className="space-y-3">
          <label 
            className="text-lg font-bold text-[#DFFF00] drop-shadow-md block"
            style={{ fontFamily: 'Comic Sans MS, cursive, sans-serif' }}
          >
{"Describe your magic audio effects 🎵"}
          </label>
          <Textarea
            placeholder="Add cinematic thunder effects with rain ambience during outdoor scenes..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="min-h-[120px] resize-none text-base bg-white/90 backdrop-blur-sm border-3 border-[#FF69B4]/50 rounded-2xl focus:border-[#DFFF00] focus:ring-[#DFFF00] placeholder:text-[#8B7BC7]/60 text-[#5B4B97] font-medium"
            style={{ fontFamily: 'Comic Sans MS, cursive, sans-serif' }}
          />
        </div>

        {/* Generate Button */}
        <Button 
          size="lg" 
          className={cn(
            "w-full h-16 text-xl font-bold rounded-full transition-all duration-300",
            "bg-gradient-to-r from-[#FF69B4] to-[#DFFF00] hover:from-[#FF69B4] hover:to-[#BFDF00]",
            "text-[#5B4B97] shadow-lg hover:shadow-xl hover:scale-[1.02]",
            "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
          )}
          onClick={handleGenerate} 
          disabled={!videoFile || !prompt}
          style={{ fontFamily: 'Comic Sans MS, cursive, sans-serif' }}
        >
          <Sparkles className="mr-3 h-6 w-6" />
          {"Generate the magic ✨"}
        </Button>        
      
      </div>
    </div>
  )
}
