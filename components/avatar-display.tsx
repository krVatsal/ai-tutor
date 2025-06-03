"use client";

import { useState, useEffect, useRef } from "react";
import ReactPlayer from 'react-player';
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { generateVideoResponse, getVideoStatus } from "@/lib/tavus";

interface AvatarDisplayProps {
  isProcessing: boolean;
  lastMessage: string;
  videoEnabled: boolean;
}

export function AvatarDisplay({ isProcessing, lastMessage, videoEnabled }: AvatarDisplayProps) {
  const [speaking, setSpeaking] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const statusCheckInterval = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (!isProcessing && lastMessage && videoEnabled) {
      generateVideo(lastMessage);
    }
  }, [isProcessing, lastMessage, videoEnabled]);

  const generateVideo = async (text: string) => {
    try {
      setIsGeneratingVideo(true);
      const response = await generateVideoResponse(text);
      const videoId = response.id;

      // Poll for video status
      statusCheckInterval.current = setInterval(async () => {
        const status = await getVideoStatus(videoId);
        if (status.status === 'completed') {
          setVideoUrl(status.url);
          setSpeaking(true);
          clearInterval(statusCheckInterval.current);
          setIsGeneratingVideo(false);
        }
      }, 2000);

    } catch (error) {
      console.error('Error generating video:', error);
      setIsGeneratingVideo(false);
    }
  };

  useEffect(() => {
    return () => {
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current);
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="relative w-full max-w-[400px] aspect-video rounded-lg overflow-hidden border">
        {(isProcessing || isGeneratingVideo) ? (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
          </div>
        ) : null}
        
        <div className="relative w-full h-full">
          {videoUrl ? (
            <ReactPlayer
              url={videoUrl}
              width="100%"
              height="100%"
              playing={speaking}
              onEnded={() => {
                setSpeaking(false);
                setVideoUrl(null);
              }}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-primary/5 to-primary/10">
              <div className={cn(
                "absolute bottom-0 left-0 right-0 h-1 bg-primary scale-x-0 origin-left transition-transform",
                speaking && "animate-[grow_2s_ease-in-out_infinite]"
              )} />
            </div>
          )}
        </div>
      </div>
      
      <div className="mt-4 text-center max-w-lg">
        <p className="text-sm text-muted-foreground">
          {isGeneratingVideo 
            ? "Generating video response..." 
            : speaking 
              ? "Mira is speaking..." 
              : "Mira is waiting for your question..."}
        </p>
        {!videoEnabled && (
          <p className="text-xs text-muted-foreground mt-1">
            Video chat time limit reached. Continuing in text mode.
          </p>
        )}
      </div>
    </div>
  );
}