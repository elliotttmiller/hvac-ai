'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { CheckCircle, XCircle, AlertCircle, Settings, ExternalLink } from 'lucide-react';

export default function AIServiceStatus() {
  const [isLoading, setIsLoading] = useState(true);
  const [showSetupInstructions, setShowSetupInstructions] = useState(false);

  useEffect(() => {
    // Placeholder for custom AI service status check
    setIsLoading(false);
  }, []);

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              AI Service Configuration
            </CardTitle>
            <CardDescription>
              Custom AI services are being integrated.
            </CardDescription>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Custom AI services are under development. Please check back later.
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}
