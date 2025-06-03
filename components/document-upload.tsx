"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { FileUp, File, FileText, FileImage, Upload, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";

interface DocumentUploadProps {
  onUpload: (file: File) => void;
  isUploading: boolean;
}

export function DocumentUpload({ onUpload, isUploading }: DocumentUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const { toast } = useToast();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.length > 0) {
      const selectedFile = acceptedFiles[0];
      const maxSize = 10 * 1024 * 1024; // 10MB limit
      
      if (selectedFile.size > maxSize) {
        toast({
          title: "File too large",
          description: "Please select a file under 10MB",
          variant: "destructive",
        });
        return;
      }
      
      setFile(selectedFile);
      setUploadProgress(0);
    }
  }, [toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    maxFiles: 1,
    disabled: isUploading
  });

  const handleUpload = async () => {
    if (file) {
      try {
        const progressInterval = setInterval(() => {
          setUploadProgress((prev) => {
            if (prev >= 90) return prev;
            return prev + 10;
          });
        }, 500);

        await onUpload(file);
        
        clearInterval(progressInterval);
        setUploadProgress(100);
        
        setTimeout(() => {
          setFile(null);
          setUploadProgress(0);
        }, 1000);
      } catch (error) {
        toast({
          title: "Upload failed",
          description: "There was an error uploading your document. Please try again.",
          variant: "destructive",
        });
        setUploadProgress(0);
      }
    }
  };

  const getFileIcon = (file: File) => {
    const type = file.type;
    if (type.includes('pdf')) return <File className="h-10 w-10 text-red-500" />;
    if (type.includes('docx') || type.includes('word')) return <FileText className="h-10 w-10 text-blue-500" />;
    if (type.includes('text')) return <FileText className="h-10 w-10 text-gray-500" />;
    return <FileImage className="h-10 w-10 text-purple-500" />;
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-2xl">Upload a Document</CardTitle>
        <CardDescription>
          Upload a PDF, DOCX, or TXT file (max 10MB) to start learning with Mira
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          {...getRootProps()}
          className={cn(
            "border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors",
            isDragActive 
              ? "border-primary/70 bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-accent",
            isUploading && "pointer-events-none opacity-70"
          )}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-2">
            <FileUp className="h-10 w-10 text-muted-foreground mb-2" />
            {isDragActive ? (
              <p className="text-lg font-medium">Drop the file here</p>
            ) : (
              <p className="text-lg font-medium">
                Drag & drop a file here, or click to select
              </p>
            )}
            <p className="text-sm text-muted-foreground">
              Supports PDF, DOCX, and TXT files up to 10MB
            </p>
          </div>
        </div>

        {file && !isUploading && (
          <div className="mt-6 flex items-center gap-4 p-4 border rounded-lg">
            {getFileIcon(file)}
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{file.name}</p>
              <p className="text-sm text-muted-foreground">
                {(file.size / 1024 / 1024).toFixed(1)} MB
              </p>
            </div>
          </div>
        )}

        {isUploading && (
          <div className="mt-6">
            <div className="flex justify-between text-sm mb-1">
              <span>Uploading {file?.name}</span>
              <span>{uploadProgress}%</span>
            </div>
            <Progress value={uploadProgress} className="h-2" />
          </div>
        )}
      </CardContent>
      <CardFooter>
        <Button 
          onClick={handleUpload} 
          disabled={!file || isUploading} 
          className="w-full"
        >
          {isUploading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              Upload Document
              <Upload className="ml-2 h-4 w-4" />
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}