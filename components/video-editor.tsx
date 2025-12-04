"use client"

import type React from "react"
import { Sparkles } from "lucide-react" // Import Sparkles component

import { useState, useRef, useEffect } from "react"
import {
  Upload,
  Play,
  Pause,
  Volume2,
  VolumeX,
  Download,
  Scissors,
  Wand2,
  Eye,
  EyeOff,
  Lock,
  Unlock,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Slider } from "@/components/ui/slider"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

type EditorState = "upload" | "processing" | "editing"

interface AudioBlock {
  id: string
  name: string
  start: number
  duration: number
  volume: number
  audioUrl?: string
}

interface Track {
  id: string
  name: string
  type: "video" | "audio" | "effects"
  visible: boolean
  locked: boolean
  volume: number
  blocks?: AudioBlock[]
}

export default function VideoEditor() {
  const [state, setState] = useState<EditorState>("upload")
  const [prompt, setPrompt] = useState("")
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(120)
  const [zoom, setZoom] = useState([1])
  const [showControls, setShowControls] = useState(true)
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const controlsTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const audioRefs = useRef<Map<string, HTMLAudioElement>>(new Map())
  
  // Créer l'URL de la vidéo quand le fichier change
  useEffect(() => {
    if (videoFile) {
      const url = URL.createObjectURL(videoFile)
      setVideoUrl(url)
      return () => URL.revokeObjectURL(url)
    }
  }, [videoFile])
  
  // Synchroniser l'état de lecture avec l'élément vidéo
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    
    if (isPlaying) {
      video.play()
    } else {
      video.pause()
    }
  }, [isPlaying])
  
  // Synchroniser le mute avec l'élément vidéo
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    video.muted = isMuted
  }, [isMuted])
  
  // Gérer les événements de la vidéo
  const handleVideoTimeUpdate = () => {
    const video = videoRef.current
    if (video) {
      setCurrentTime(video.currentTime)
    }
  }
  
  const handleVideoLoadedMetadata = () => {
    const video = videoRef.current
    if (video) {
      setDuration(video.duration)
    }
  }
  
  const handleVideoEnded = () => {
    setIsPlaying(false)
  }
  
  const handleSeek = (value: number[]) => {
    const video = videoRef.current
    if (video) {
      video.currentTime = value[0]
      setCurrentTime(value[0])
    }
  }
  
  // Gestion de l'affichage/masquage des contrôles
  const resetControlsTimeout = () => {
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current)
    }
    setShowControls(true)
    controlsTimeoutRef.current = setTimeout(() => {
      if (isPlaying) {
        setShowControls(false)
      }
    }, 2500)
  }
  
  const handleVideoMouseMove = () => {
    resetControlsTimeout()
  }
  
  const handleVideoMouseLeave = () => {
    if (isPlaying) {
      controlsTimeoutRef.current = setTimeout(() => {
        setShowControls(false)
      }, 1000)
    }
  }
  
  // Réinitialiser le timeout quand la vidéo est mise en pause
  useEffect(() => {
    if (!isPlaying) {
      setShowControls(true)
      if (controlsTimeoutRef.current) {
        clearTimeout(controlsTimeoutRef.current)
      }
    } else {
      resetControlsTimeout()
    }
  }, [isPlaying])
  
  // Nettoyer le timeout au démontage
  useEffect(() => {
    return () => {
      if (controlsTimeoutRef.current) {
        clearTimeout(controlsTimeoutRef.current)
      }
    }
  }, [])

  const [tracks, setTracks] = useState<Track[]>([
    {
      id: "track-1",
      name: "Video + Original Audio",
      type: "video",
      visible: true,
      locked: false,
      volume: 80,
    },
    {
      id: "track-2",
      name: "Noise Effects",
      type: "effects",
      visible: true,
      locked: false,
      volume: 60,
      blocks: [
        { 
          id: "block-1", 
          name: "Thunder", 
          start: 10, 
          duration: 15, 
          volume: 70,
          audioUrl: "https://soundbible.com/wav/Thunder-Mike_Koenig-315681025.wav"
        },
        { id: "block-2", name: "Rain Ambience", start: 30, duration: 45, volume: 50 },
        { id: "block-3", name: "Wind", start: 80, duration: 25, volume: 60 },
      ],
    },
  ])

  const [selectedBlock, setSelectedBlock] = useState<string | null>(null)
  
  // États pour le drag & drop des blocs
  const [draggingBlock, setDraggingBlock] = useState<string | null>(null)
  const [dragStartX, setDragStartX] = useState<number>(0)
  const [dragStartTime, setDragStartTime] = useState<number>(0)
  const timelineRef = useRef<HTMLDivElement>(null)
  
  // États pour le redimensionnement des blocs
  const [resizingBlock, setResizingBlock] = useState<string | null>(null)
  const [resizeEdge, setResizeEdge] = useState<"left" | "right" | null>(null)
  const [resizeStartX, setResizeStartX] = useState<number>(0)
  const [resizeStartData, setResizeStartData] = useState<{ start: number; duration: number } | null>(null)

  // Initialiser les éléments audio pour les blocs avec audioUrl
  useEffect(() => {
    tracks.forEach((track) => {
      if (track.blocks) {
        track.blocks.forEach((block) => {
          if (block.audioUrl && !audioRefs.current.has(block.id)) {
            const audio = new Audio(block.audioUrl)
            audio.preload = "auto"
            audio.volume = (block.volume / 100) * (track.volume / 100)
            audioRefs.current.set(block.id, audio)
          }
        })
      }
    })
    
    // Nettoyer les audios au démontage
    return () => {
      audioRefs.current.forEach((audio) => {
        audio.pause()
        audio.src = ""
      })
      audioRefs.current.clear()
    }
  }, [tracks])
  
  // Gérer la lecture des blocs audio en fonction du temps de la vidéo
  useEffect(() => {
    tracks.forEach((track) => {
      if (track.blocks && track.visible && !track.locked) {
        track.blocks.forEach((block) => {
          const audio = audioRefs.current.get(block.id)
          if (audio) {
            const blockEnd = block.start + block.duration
            const isInRange = currentTime >= block.start && currentTime < blockEnd
            
            if (isPlaying && isInRange && !isMuted) {
              if (audio.paused) {
                // Synchroniser la position de l'audio avec le temps de la vidéo
                const audioTime = currentTime - block.start
                if (Math.abs(audio.currentTime - audioTime) > 0.5) {
                  audio.currentTime = audioTime
                }
                audio.play().catch(() => {})
              }
            } else {
              if (!audio.paused) {
                audio.pause()
              }
            }
            
            // Mettre à jour le volume
            audio.volume = (block.volume / 100) * (track.volume / 100)
            audio.muted = isMuted
          }
        })
      }
    })
  }, [currentTime, isPlaying, isMuted, tracks])

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
        alert('Fichier uploadé avec succès!')
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
        alert('Fichier uploadé avec succès!')
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
          setState("editing")
        }
      }, 100)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const toggleTrackVisibility = (trackId: string) => {
    setTracks(tracks.map((t) => (t.id === trackId ? { ...t, visible: !t.visible } : t)))
  }

  const toggleTrackLock = (trackId: string) => {
    setTracks(tracks.map((t) => (t.id === trackId ? { ...t, locked: !t.locked } : t)))
  }

  const updateTrackVolume = (trackId: string, volume: number) => {
    setTracks(tracks.map((t) => (t.id === trackId ? { ...t, volume } : t)))
  }
  
  // Fonction pour mettre à jour la position d'un bloc
  const updateBlockStart = (trackId: string, blockId: string, newStart: number) => {
    setTracks(tracks.map((track) => {
      if (track.id === trackId && track.blocks) {
        return {
          ...track,
          blocks: track.blocks.map((block) => {
            if (block.id === blockId) {
              // S'assurer que le bloc reste dans les limites de la timeline
              const clampedStart = Math.max(0, Math.min(newStart, duration - block.duration))
              return { ...block, start: clampedStart }
            }
            return block
          })
        }
      }
      return track
    }))
  }
  
  // Fonction pour mettre à jour le début et la durée d'un bloc (pour le redimensionnement)
  const updateBlockDimensions = (trackId: string, blockId: string, newStart: number, newDuration: number) => {
    setTracks(tracks.map((track) => {
      if (track.id === trackId && track.blocks) {
        return {
          ...track,
          blocks: track.blocks.map((block) => {
            if (block.id === blockId) {
              // S'assurer que la durée minimale est de 1 seconde
              const clampedDuration = Math.max(1, newDuration)
              // S'assurer que le bloc reste dans les limites de la timeline
              const clampedStart = Math.max(0, Math.min(newStart, duration - clampedDuration))
              return { ...block, start: clampedStart, duration: clampedDuration }
            }
            return block
          })
        }
      }
      return track
    }))
  }
  
  // Gestionnaires de redimensionnement pour les blocs
  const handleResizeMouseDown = (e: React.MouseEvent, trackId: string, block: AudioBlock, edge: "left" | "right") => {
    e.preventDefault()
    e.stopPropagation()
    
    // Vérifier si la piste est verrouillée
    const track = tracks.find(t => t.id === trackId)
    if (track?.locked) return
    
    setResizingBlock(block.id)
    setResizeEdge(edge)
    setResizeStartX(e.clientX)
    setResizeStartData({ start: block.start, duration: block.duration })
    setSelectedBlock(block.id)
  }
  
  // Gestionnaires de drag & drop pour les blocs
  const handleBlockMouseDown = (e: React.MouseEvent, trackId: string, block: AudioBlock) => {
    e.preventDefault()
    e.stopPropagation()
    
    // Vérifier si la piste est verrouillée
    const track = tracks.find(t => t.id === trackId)
    if (track?.locked) return
    
    setDraggingBlock(block.id)
    setDragStartX(e.clientX)
    setDragStartTime(block.start)
    setSelectedBlock(block.id)
  }
  
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!draggingBlock || !timelineRef.current) return
    
    const deltaX = e.clientX - dragStartX
    const timelineWidth = timelineRef.current.offsetWidth
    const pixelsPerSecond = (100 * 12 * zoom[0]) / duration
    const deltaTime = deltaX / pixelsPerSecond
    
    const newStart = dragStartTime + deltaTime
    
    // Trouver la piste contenant le bloc
    const trackWithBlock = tracks.find(t => t.blocks?.some(b => b.id === draggingBlock))
    if (trackWithBlock) {
      updateBlockStart(trackWithBlock.id, draggingBlock, newStart)
    }
  }
  
  const handleMouseUp = () => {
    setDraggingBlock(null)
    setResizingBlock(null)
    setResizeEdge(null)
    setResizeStartData(null)
  }
  
  // Effet pour gérer les événements globaux de souris pendant le drag
  useEffect(() => {
    if (draggingBlock) {
      const handleGlobalMouseMove = (e: MouseEvent) => {
        if (!timelineRef.current) return
        
        const deltaX = e.clientX - dragStartX
        const pixelsPerSecond = (100 * 12 * zoom[0]) / duration
        const deltaTime = deltaX / pixelsPerSecond
        
        const newStart = dragStartTime + deltaTime
        
        const trackWithBlock = tracks.find(t => t.blocks?.some(b => b.id === draggingBlock))
        if (trackWithBlock) {
          updateBlockStart(trackWithBlock.id, draggingBlock, newStart)
        }
      }
      
      const handleGlobalMouseUp = () => {
        setDraggingBlock(null)
      }
      
      window.addEventListener('mousemove', handleGlobalMouseMove)
      window.addEventListener('mouseup', handleGlobalMouseUp)
      
      return () => {
        window.removeEventListener('mousemove', handleGlobalMouseMove)
        window.removeEventListener('mouseup', handleGlobalMouseUp)
      }
    }
  }, [draggingBlock, dragStartX, dragStartTime, zoom, duration, tracks])
  
  // Effet pour gérer les événements globaux de souris pendant le redimensionnement
  useEffect(() => {
    if (resizingBlock && resizeStartData && resizeEdge) {
      const handleGlobalMouseMove = (e: MouseEvent) => {
        if (!timelineRef.current) return
        
        const deltaX = e.clientX - resizeStartX
        const pixelsPerSecond = (100 * 12 * zoom[0]) / duration
        const deltaTime = deltaX / pixelsPerSecond
        
        const trackWithBlock = tracks.find(t => t.blocks?.some(b => b.id === resizingBlock))
        if (trackWithBlock) {
          if (resizeEdge === "left") {
            // Redimensionnement par la gauche: on déplace le début et on ajuste la durée
            const newStart = resizeStartData.start + deltaTime
            const newDuration = resizeStartData.duration - deltaTime
            updateBlockDimensions(trackWithBlock.id, resizingBlock, newStart, newDuration)
          } else {
            // Redimensionnement par la droite: on change seulement la durée
            const newDuration = resizeStartData.duration + deltaTime
            updateBlockDimensions(trackWithBlock.id, resizingBlock, resizeStartData.start, newDuration)
          }
        }
      }
      
      const handleGlobalMouseUp = () => {
        setResizingBlock(null)
        setResizeEdge(null)
        setResizeStartData(null)
      }
      
      window.addEventListener('mousemove', handleGlobalMouseMove)
      window.addEventListener('mouseup', handleGlobalMouseUp)
      
      return () => {
        window.removeEventListener('mousemove', handleGlobalMouseMove)
        window.removeEventListener('mouseup', handleGlobalMouseUp)
      }
    }
  }, [resizingBlock, resizeStartX, resizeStartData, resizeEdge, zoom, duration, tracks])

  if (state === "upload") {
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
    <div className="flex h-full flex-col bg-background">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-6 py-3 bg-card">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-muted p-2">
            <Scissors className="h-5 w-5 text-foreground" />
          </div>
          <div>
            <h2 className="text-sm font-medium">{videoFile?.name || "Untitled Video"}</h2>
            <p className="text-xs text-muted-foreground">{"Multi-track editor"}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            {"Export"}
          </Button>
        </div>
      </header>

      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Video Preview Section */}
        <div className="flex-shrink-0 border-b border-border bg-muted/30" style={{ height: "60%" }}>
          <div className="h-full flex items-center justify-center p-4">
            <div 
              className="flex flex-col w-full max-w-5xl h-full"
              onMouseMove={handleVideoMouseMove}
              onMouseLeave={handleVideoMouseLeave}
            >
              {/* Video Container */}
              <div className="flex-1 min-h-0 bg-black rounded-t-lg overflow-hidden shadow-2xl flex items-center justify-center cursor-pointer">
                {videoUrl ? (
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    className="h-full w-full object-contain"
                    onTimeUpdate={handleVideoTimeUpdate}
                    onLoadedMetadata={handleVideoLoadedMetadata}
                    onEnded={handleVideoEnded}
                    onClick={() => setIsPlaying(!isPlaying)}
                  />
                ) : (
                  <div className="text-center space-y-4">
                    <div className="mx-auto h-20 w-20 rounded-full bg-white/10 flex items-center justify-center">
                      <Play className="h-10 w-10 text-white" />
                    </div>
                    <p className="text-white/70 text-sm">{"Video Preview"}</p>
                  </div>
                )}
              </div>

              {/* Playback Controls Bar */}
              <div 
                className={cn(
                  "transition-opacity duration-300",
                  showControls ? "opacity-100" : "opacity-0 pointer-events-none"
                )}
              >
                <div className="bg-black/90 backdrop-blur-sm rounded-b-lg px-4 py-3 space-y-2">
                  {/* Timeline Scrubber */}
                  <div className="relative">
                    <Slider
                      value={[currentTime]}
                      onValueChange={handleSeek}
                      max={duration}
                      step={0.1}
                      className="cursor-pointer"
                    />
                  </div>

                  {/* Controls Row */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-8 w-8 text-white hover:bg-white/20"
                        onClick={() => setIsPlaying(!isPlaying)}
                      >
                        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                      </Button>
                      <span className="text-xs text-white/90 font-mono tabular-nums">
                        {formatTime(currentTime)} / {formatTime(duration)}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-8 w-8 text-white hover:bg-white/20"
                        onClick={() => setIsMuted(!isMuted)}
                      >
                        {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden bg-background">
          {/* Timeline Header with Controls */}
          <div className="flex items-center justify-between border-b border-border px-3 py-1.5 bg-card">
            <div className="flex items-center gap-3">
              <span className="text-xs font-medium">{"Timeline"}</span>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-muted-foreground">{"Zoom"}</span>
                <Slider value={zoom} onValueChange={setZoom} min={0.5} max={3} step={0.1} className="w-20" />
              </div>
            </div>
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="sm" className="h-7 text-xs">
                <Scissors className="mr-1.5 h-3 w-3" />
                {"Split"}
              </Button>
            </div>
          </div>

          {/* Timeline Content */}
          <div className="flex-1 flex overflow-hidden">
            {/* Track Headers */}
            <div className="w-48 border-r border-border bg-card flex-shrink-0 overflow-y-auto">
              {tracks.map((track) => (
                <div key={track.id} className="border-b border-border px-2 py-1.5 space-y-1" style={{ height: "56px" }}>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium truncate">{track.name}</span>
                    <div className="flex items-center">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-5 w-5"
                        onClick={() => toggleTrackVisibility(track.id)}
                      >
                        {track.visible ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                      </Button>
                      <Button variant="ghost" size="icon" className="h-5 w-5" onClick={() => toggleTrackLock(track.id)}>
                        {track.locked ? <Lock className="h-3 w-3" /> : <Unlock className="h-3 w-3" />}
                      </Button>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Volume2 className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                    <Slider
                      value={[track.volume]}
                      onValueChange={(value) => updateTrackVolume(track.id, value[0])}
                      max={100}
                      className="flex-1"
                    />
                    <span className="text-[10px] text-muted-foreground w-6 text-right tabular-nums">{track.volume}%</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Timeline Tracks with Ruler */}
            <div 
              ref={timelineRef}
              className="flex-1 overflow-auto relative"
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              {/* Time Ruler */}
              <div className="sticky top-0 z-10 h-6 border-b border-border bg-card flex items-center px-2">
                {Array.from({ length: Math.ceil(duration / 10) + 1 }).map((_, i) => (
                  <div key={i} className="flex-shrink-0" style={{ width: `${100 * zoom[0]}px` }}>
                    <span className="text-[10px] text-muted-foreground font-mono tabular-nums">{formatTime(i * 10)}</span>
                  </div>
                ))}
              </div>

              {/* Tracks Container */}
              <div className="relative">
                {/* Playhead */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-foreground z-20 pointer-events-none"
                  style={{ left: `${(currentTime / duration) * 100 * 12 * zoom[0]}px` }}
                >
                  <div className="w-3 h-3 bg-foreground rounded-full -ml-1.25 -mt-1.5" />
                </div>

                {/* Track Lanes */}
                {tracks.map((track, trackIndex) => (
                  <div key={track.id} className="border-b border-border" style={{ height: "56px" }}>
                    <div className="relative h-full p-1.5">
                      {track.type === "video" ? (
                        <div className="h-full bg-muted rounded relative overflow-hidden">
                          <div className="absolute inset-0 flex items-center justify-center gap-0.5 px-2">
                            {/* Simulated waveform */}
                            {Array.from({ length: 120 }).map((_, i) => {
                              const height = Math.random() * 60 + 20
                              return (
                                <div
                                  key={i}
                                  className="flex-1 bg-foreground/30 rounded-full"
                                  style={{ height: `${height}%` }}
                                />
                              )
                            })}
                          </div>
                          <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-xs font-medium text-foreground/70 bg-background/50 px-2 py-1 rounded">
                              {"Original Audio"}
                            </span>
                          </div>
                        </div>
                      ) : track.blocks ? (
                        <div className="relative h-full">
                          {track.blocks.map((block) => (
                            <div
                              key={block.id}
                              className={cn(
                                "absolute h-full rounded select-none group",
                                "bg-foreground/80 hover:bg-foreground",
                                selectedBlock === block.id && "ring-2 ring-ring ring-offset-2 ring-offset-background",
                                (draggingBlock === block.id || resizingBlock === block.id) && "opacity-90 shadow-lg z-10",
                                track.locked && "opacity-60",
                              )}
                              style={{
                                left: `${(block.start / duration) * 100 * 12 * zoom[0]}px`,
                                width: `${(block.duration / duration) * 100 * 12 * zoom[0]}px`,
                                transition: (draggingBlock === block.id || resizingBlock === block.id) ? 'none' : 'all 0.15s ease-out',
                              }}
                              onClick={(e) => {
                                e.stopPropagation()
                                if (!draggingBlock && !resizingBlock) setSelectedBlock(block.id)
                              }}
                            >
                              {/* Poignée de redimensionnement gauche */}
                              <div
                                className={cn(
                                  "absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize z-20",
                                  "bg-background/0 hover:bg-background/30 transition-colors",
                                  "rounded-l flex items-center justify-center",
                                  track.locked && "cursor-not-allowed"
                                )}
                                onMouseDown={(e) => handleResizeMouseDown(e, track.id, block, "left")}
                              >
                                <div className="w-0.5 h-4 bg-background/50 rounded opacity-0 group-hover:opacity-100 transition-opacity" />
                              </div>
                              
                              {/* Zone centrale draggable */}
                              <div 
                                className={cn(
                                  "absolute left-2 right-2 top-0 bottom-0 cursor-move",
                                  track.locked && "cursor-not-allowed"
                                )}
                                onMouseDown={(e) => handleBlockMouseDown(e, track.id, block)}
                              >
                                <div className="h-full p-2 flex flex-col justify-between pointer-events-none">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs font-medium text-background truncate">{block.name}</span>
                                    <span className="text-[10px] text-background/70 font-mono">{formatTime(block.duration)}s</span>
                                  </div>
                                  <div className="flex items-center gap-1">
                                    {/* Simulated audio block waveform */}
                                    {Array.from({ length: 20 }).map((_, i) => {
                                      const height = Math.random() * 50 + 30
                                      return (
                                        <div
                                          key={i}
                                          className="flex-1 bg-background/40 rounded-full"
                                          style={{ height: `${height}%` }}
                                        />
                                      )
                                    })}
                                  </div>
                                </div>
                              </div>
                              
                              {/* Poignée de redimensionnement droite */}
                              <div
                                className={cn(
                                  "absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize z-20",
                                  "bg-background/0 hover:bg-background/30 transition-colors",
                                  "rounded-r flex items-center justify-center",
                                  track.locked && "cursor-not-allowed"
                                )}
                                onMouseDown={(e) => handleResizeMouseDown(e, track.id, block, "right")}
                              >
                                <div className="w-0.5 h-4 bg-background/50 rounded opacity-0 group-hover:opacity-100 transition-opacity" />
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {selectedBlock && (
        <div className="absolute right-0 top-0 bottom-0 w-80 border-l border-border bg-card shadow-2xl p-4 space-y-4 overflow-y-auto">
          <div className="flex items-center justify-between">
            <h3 className="font-medium">{"Block Properties"}</h3>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setSelectedBlock(null)}>
              <Scissors className="h-4 w-4" />
            </Button>
          </div>

          {tracks
            .find((t) => t.blocks?.some((b) => b.id === selectedBlock))
            ?.blocks?.find((b) => b.id === selectedBlock) && (
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">{"Block Name"}</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 rounded-md border border-input bg-background text-sm"
                  defaultValue={
                    tracks
                      .find((t) => t.blocks?.some((b) => b.id === selectedBlock))
                      ?.blocks?.find((b) => b.id === selectedBlock)?.name
                  }
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">{"Volume"}</label>
                <Slider
                  defaultValue={[
                    tracks
                      .find((t) => t.blocks?.some((b) => b.id === selectedBlock))
                      ?.blocks?.find((b) => b.id === selectedBlock)?.volume || 50,
                  ]}
                  max={100}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <label className="text-sm font-medium">{"Start"}</label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 rounded-md border border-input bg-background text-sm font-mono"
                    defaultValue={formatTime(
                      tracks
                        .find((t) => t.blocks?.some((b) => b.id === selectedBlock))
                        ?.blocks?.find((b) => b.id === selectedBlock)?.start || 0,
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">{"Duration"}</label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 rounded-md border border-input bg-background text-sm font-mono"
                    defaultValue={formatTime(
                      tracks
                        .find((t) => t.blocks?.some((b) => b.id === selectedBlock))
                        ?.blocks?.find((b) => b.id === selectedBlock)?.duration || 0,
                    )}
                  />
                </div>
              </div>

              <div className="pt-2 space-y-2">
                <Button variant="outline" size="sm" className="w-full bg-transparent">
                  <Scissors className="mr-2 h-4 w-4" />
                  {"Split Block"}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full text-destructive hover:text-destructive bg-transparent"
                >
                  {"Delete Block"}
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
