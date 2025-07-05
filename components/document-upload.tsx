"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { FileUp, File, Loader2, Upload } from "lucide-react";
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

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    // Handle rejected files first
    if (rejectedFiles?.length > 0) {
      const rejection = rejectedFiles[0];
      if (rejection.errors?.[0]?.code === 'file-too-large') {
        toast({
          title: "File too large",
          description: "Please select a PDF file under 10MB",
          variant: "destructive",
        });
      } else if (rejection.errors?.[0]?.code === 'file-invalid-type') {
        toast({
          title: "Invalid file type",
          description: "Please select a PDF file (.pdf extension required)",
          variant: "destructive",
        });
      } else {
        toast({
          title: "File rejected",
          description: "Please select a valid PDF file under 10MB",
          variant: "destructive",
        });
      }
      return;
    }

    if (acceptedFiles?.length > 0) {
      const selectedFile = acceptedFiles[0];
      const maxSize = 10 * 1024 * 1024; // 10MB limit
      
      // Double-check file size
      if (selectedFile.size > maxSize) {
        const fileSizeMB = (selectedFile.size / 1024 / 1024).toFixed(1);
        toast({
          title: "File too large",
          description: `File size (${fileSizeMB}MB) exceeds the 10MB limit. Please select a smaller file.`,
          variant: "destructive",
        });
        return;
      }
      
      // Double-check file type
      if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
        const fileType = selectedFile.name.split('.').pop()?.toUpperCase() || 'unknown';
        toast({
          title: "Invalid file type",
          description: `${fileType} files are not supported. Please select a PDF file.`,
          variant: "destructive",
        });
        return;
      }
      
      // Check if file appears to be empty
      if (selectedFile.size < 1024) { // Less than 1KB
        toast({
          title: "File too small",
          description: "The selected file appears to be empty or corrupted. Please select a different file.",
          variant: "destructive",
        });
        return;
      }
      
      setFile(selectedFile);
      setUploadProgress(0);
      
      toast({
        title: "File selected",
        description: `${selectedFile.name} is ready for upload`,
      });
    }
  }, [toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
    disabled: isUploading,
    onError: (error) => {
      console.error('Dropzone error:', error);
      toast({
        title: "Upload error",
        description: "There was an error with the file selection. Please try again.",
        variant: "destructive",
      });
    }
  });

  const handleUpload = async () => {
    if (file) {
      try {
        // Start progress simulation
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

        toast({
          title: "Upload successful",
          description: "Your document has been processed and vectorized successfully",
        });
      } catch (error: any) {
        console.error('Upload error:', error);
        
        // Clear progress
        setUploadProgress(0);
        
        // Handle different types of errors
        let errorTitle = "Upload failed";
        let errorDescription = "There was an error processing your document. Please try again.";
        
        if (error?.response?.data?.detail) {
          const errorDetail = error.response.data.detail;
          
          // Handle structured error responses
          if (typeof errorDetail === 'object' && errorDetail.message) {
            errorTitle = errorDetail.error || "Upload failed";
            errorDescription = errorDetail.message;
            
            // Add specific guidance based on error code
            switch (errorDetail.code) {
              case "INVALID_FILE_TYPE":
                errorDescription += " Make sure your file has a .pdf extension.";
                break;
              case "FILE_TOO_LARGE":
                errorDescription += " Try compressing your PDF or splitting it into smaller files.";
                break;
              case "INVALID_PDF":
                errorDescription += " Try saving your document as a new PDF or use a different PDF viewer to re-save it.";
                break;
              case "NO_TEXT_CONTENT":
                errorDescription += " PDFs with only images are not supported. Please use a PDF with selectable text.";
                break;
              case "DOCUMENT_TOO_COMPLEX":
                errorDescription += " Try removing images or reducing the document size.";
                break;
              case "PROCESSING_ERROR":
                errorDescription += " This might be a temporary issue. Please try again in a few moments.";
                break;
            }
          } else if (typeof errorDetail === 'string') {
            errorDescription = errorDetail;
          }
        } else if (error?.message) {
          errorDescription = error.message;
        } else if (error?.response?.status === 413) {
          errorTitle = "File too large";
          errorDescription = "The file size exceeds the server limit. Please upload a smaller file.";
        } else if (error?.response?.status === 401) {
          errorTitle = "Authentication required";
          errorDescription = "Please sign in and try again.";
        } else if (error?.response?.status >= 500) {
          errorTitle = "Server error";
          errorDescription = "There was a server error. Please try again later or contact support.";
        }

        toast({
          title: errorTitle,
          description: errorDescription,
          variant: "destructive",
          duration: 8000, // Show error longer
        });
      }
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-2xl">Upload a Document</CardTitle>
        <CardDescription>
          Upload a PDF file (max 10MB) to start learning with Mira
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
              <p className="text-lg font-medium">Drop the PDF here</p>
            ) : (
              <p className="text-lg font-medium">
                Drag & drop a PDF file here, or click to select
              </p>
            )}
            <p className="text-sm text-muted-foreground">
              Supports PDF files up to 10MB
            </p>
          </div>
        </div>

        {file && !isUploading && (
          <div className="mt-6 flex items-center gap-4 p-4 border rounded-lg">
            <File className="h-10 w-10 text-red-500" />
            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{file.name}</p>
              <p className="text-sm text-muted-foreground">
                {(file.size / 1024 / 1024).toFixed(1)} MB
              </p>
            </div>
          </div>
        )}

        {isUploading && (
          <div className="mt-6 space-y-3">
            <div className="flex items-center gap-3 p-4 border rounded-lg bg-blue-50 dark:bg-blue-950/20">
              <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
              <div className="flex-1">
                <p className="font-medium text-blue-900 dark:text-blue-100">Processing {file?.name}</p>
                <p className="text-sm text-blue-600 dark:text-blue-300">
                  {uploadProgress < 30 && "Uploading file..."}
                  {uploadProgress >= 30 && uploadProgress < 60 && "Extracting text content..."}
                  {uploadProgress >= 60 && uploadProgress < 90 && "Creating vector database..."}
                  {uploadProgress >= 90 && "Finalizing..."}
                </p>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="h-2" />
            </div>
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