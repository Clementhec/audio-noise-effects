"use client"

import { useState } from "react"
import VideoUpload from "@/components/video-upload"
import VideoEditorPage from "@/components/video-editor-page"

interface AudioBlock {
  id: string
  name: string
  start: number
  duration: number
  volume: number
  audioUrl?: string
}

export default function VideoEditor() {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [soundEffects, setSoundEffects] = useState<AudioBlock[]>([])

  const handleUploadComplete = (file: File, url: string, effects: AudioBlock[]) => {
    setVideoFile(file)
    setVideoUrl(url)
    setSoundEffects(effects)
  }

  if (videoFile && videoUrl) {
    return <VideoEditorPage videoFile={videoFile} videoUrl={videoUrl} soundEffects={soundEffects} />
  }

  return <VideoUpload onComplete={handleUploadComplete} />
}
