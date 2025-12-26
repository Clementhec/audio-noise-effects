"use client"

import type React from "react"
import { Sparkles } from "lucide-react"
import { useState } from "react"
import { Upload, Wand2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

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
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="w-full max-w-md space-y-6 text-center">
          <div className="flex justify-center">
            <div className="relative">
              <div className="h-20 w-20 rounded-full bg-primary/10 animate-pulse" />
              <Sparkles className="absolute inset-0 m-auto h-10 w-10 text-primary animate-pulse" />
            </div>
          </div>
          <div className="space-y-3">
            <h2 className="text-2xl font-medium">{"Processing your video"}</h2>
            <p className="text-muted-foreground">{"Applying noise effects and audio enhancements..."}</p>
          </div>
          <div className="space-y-2">
            <Progress value={progress} className="h-2" />
            <p className="text-sm text-muted-foreground">{progress}%</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col items-center justify-center p-8">
      <div className="w-full max-w-3xl space-y-8">
        {/* Header */}
        <div className="text-center space-y-3">
          <div className="flex justify-center">
            <div className="rounded-full bg-muted p-4">
              <Wand2 className="h-8 w-8 text-foreground" />
            </div>
          </div>
          <h1 className="text-4xl font-medium tracking-tight text-balance">{"Create Enhanced Videos"}</h1>
          <p className="text-lg text-muted-foreground text-balance">
            {"Upload your video and describe the audio effects you want to apply"}
          </p>
        </div>

        {/* Upload Area */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className={cn(
            "border-2 border-dashed rounded-xl p-12 transition-all duration-200",
            videoFile ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-muted/30",
          )}
        >
          <div className="flex flex-col items-center gap-4 text-center">
            <div className="rounded-full bg-muted p-4">
              <Upload className="h-8 w-8 text-muted-foreground" />
            </div>
            <div className="space-y-2">
              <p className="text-base font-medium">{videoFile ? videoFile.name : "Drop your video here"}</p>
              <p className="text-sm text-muted-foreground">{"or click to browse • MP4, MOV, AVI • Max 500MB"}</p>
            </div>
            <input type="file" accept="video/*" onChange={handleFileUpload} className="hidden" id="video-upload" />
            <label htmlFor="video-upload">
              <Button variant="secondary" size="lg" asChild>
                <span>{"Choose File"}</span>
              </Button>
            </label>
          </div>
        </div>

        {/* Prompt Area */}
        <div className="space-y-3">
          <label className="text-sm font-medium">{"Describe your audio effects"}</label>
          <Textarea
            placeholder="Add cinematic thunder sound effects with rain ambience during outdoor scenes..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="min-h-[120px] resize-none text-base"
          />
        </div>

        {/* Generate Button */}
        <Button size="lg" className="w-full h-12 text-base" onClick={handleGenerate} disabled={!videoFile || !prompt}>
          <Sparkles className="mr-2 h-5 w-5" />
          {"Generate Enhanced Video"}
        </Button>
      </div>
    </div>
  )
}


