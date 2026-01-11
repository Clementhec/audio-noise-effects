"use client"

import { useState } from "react"
import VideoUpload from "@/components/video-upload"
import VideoEditorPage from "@/components/video-editor-page"

export default function VideoEditor() {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)

  const handleUploadComplete = (file: File, url: string) => {
    setVideoFile(file)
    setVideoUrl(url)
  }

  if (videoFile && videoUrl) {
    return <VideoEditorPage videoFile={videoFile} videoUrl={videoUrl} />
  }

  return <VideoUpload onComplete={handleUploadComplete} />
}
