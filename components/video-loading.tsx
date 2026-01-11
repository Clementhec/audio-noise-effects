"use client"

import Image from "next/image"
import { Progress } from "@/components/ui/progress"

interface VideoLoadingProps {
  progress: number
}

export default function VideoLoading({ progress }: VideoLoadingProps) {
  return (
    <div className="flex h-full items-center justify-center p-8 bg-gradient-to-br from-[#8B7BC7] via-[#A594D8] to-[#9B8BD3] relative overflow-hidden">
      {/* Blob pattern background */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-10 left-10 w-32 h-32 bg-[#7B6BB7] rounded-full blur-2xl animate-pulse" />
        <div className="absolute top-40 right-20 w-40 h-40 bg-[#9B8BD3] rounded-full blur-2xl animate-pulse delay-300" />
        <div className="absolute bottom-20 left-1/4 w-36 h-36 bg-[#8B7BC7] rounded-full blur-2xl animate-pulse delay-500" />
        <div className="absolute bottom-10 right-10 w-28 h-28 bg-[#A594D8] rounded-full blur-2xl animate-pulse delay-700" />
      </div>
      
      <div className="w-full max-w-md space-y-6 text-center relative z-10">
        <div className="flex justify-center">
          <div className="relative">
            <Image 
              src="/soundeasy.gif" 
              alt="SoundEasy Loading" 
              width={400} 
              height={200}
              className="drop-shadow-2xl"
              unoptimized
            />
          </div>
        </div>
        <div className="space-y-3">
          <h2 className="text-2xl font-bold text-[#DFFF00] drop-shadow-[0_2px_4px_rgba(0,0,0,0.3)]" style={{ fontFamily: 'Comic Sans MS, cursive, sans-serif' }}>
            {"Processing..."}
          </h2>
          <p className="text-white/90 font-medium drop-shadow-md">
            {"Applying magic audio effects ✨"}
          </p>
        </div>
        <div className="space-y-2">
          <div className="relative">
            <Progress 
              value={progress} 
              className="h-4 bg-[#7B6BB7]/50 border-2 border-[#DFFF00] rounded-full overflow-hidden [&>div]:bg-gradient-to-r [&>div]:from-[#FF69B4] [&>div]:to-[#DFFF00]" 
            />
          </div>
          <p className="text-lg font-bold text-[#DFFF00] drop-shadow-md">{progress}%</p>
        </div>
      </div>
    </div>
  )
}
