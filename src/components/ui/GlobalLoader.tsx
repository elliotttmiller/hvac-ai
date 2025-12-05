"use client";

import { Loader2, Zap } from "lucide-react";

export default function GlobalLoader({ message }: { message?: string }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/60 backdrop-blur-sm">
      <div className="text-center space-y-4 px-4">
        <div className="mx-auto w-14 h-14 bg-primary rounded-lg flex items-center justify-center animate-pulse">
          <Zap className="h-8 w-8 text-primary-foreground" />
        </div>
        <div className="space-y-2">
          <Loader2 className="mx-auto h-6 w-6 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">{message ?? "Loading..."}</p>
        </div>
      </div>
    </div>
  );
}
