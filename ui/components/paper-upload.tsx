"use client";

import { useCallback, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FileUp, X, FileText } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface UploadedPaper {
  name: string;
  size: number;
}

export function PaperUpload() {
  const [papers, setPapers] = useState<UploadedPaper[]>([]);
  const [dragging, setDragging] = useState(false);

  const uploadFiles = useCallback(async (files: FileList) => {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const res = await fetch(`${API}/api/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setPapers((prev) => [
        ...prev,
        ...data.uploaded.map((u: any) => ({ name: u.name, size: u.size })),
      ]);
    } catch {
      // ignore
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (e.dataTransfer.files.length > 0) {
        uploadFiles(e.dataTransfer.files);
      }
    },
    [uploadFiles]
  );

  const removePaper = (name: string) => {
    setPapers((prev) => prev.filter((p) => p.name !== name));
  };

  return (
    <Card className="border-border/50 bg-card">
      <CardContent className="pt-6">
        <label className="text-sm font-medium text-foreground mb-2 block">
          Papers
        </label>

        {/* Drop zone */}
        <div
          className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
            dragging
              ? "border-blue-400 bg-blue-500/10"
              : "border-border/50 hover:border-muted-foreground/30"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => {
            const input = document.createElement("input");
            input.type = "file";
            input.multiple = true;
            input.accept = ".pdf,.md,.txt,.tex";
            input.onchange = (e) => {
              const files = (e.target as HTMLInputElement).files;
              if (files) uploadFiles(files);
            };
            input.click();
          }}
        >
          <FileUp className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">
            Drop PDF, MD, or TXT files here
          </p>
        </div>

        {/* Uploaded files */}
        {papers.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {papers.map((p) => (
              <Badge
                key={p.name}
                variant="secondary"
                className="gap-1.5 pl-2 pr-1 py-1"
              >
                <FileText className="h-3 w-3" />
                {p.name}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removePaper(p.name);
                  }}
                  className="ml-1 rounded-full p-0.5 hover:bg-muted-foreground/20"
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
