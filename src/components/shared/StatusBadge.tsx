'use client';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

export type StatusType = 'success' | 'error' | 'warning' | 'info' | 'pending' | 'processing';

interface StatusBadgeProps {
  status: StatusType;
  children: React.ReactNode;
  className?: string;
}

const statusConfig = {
  success: {
    variant: 'default' as const,
    className: 'bg-green-100 text-green-800 border-green-200',
  },
  error: {
    variant: 'destructive' as const,
    className: 'bg-red-100 text-red-800 border-red-200',
  },
  warning: {
    variant: 'secondary' as const,
    className: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  },
  info: {
    variant: 'secondary' as const,
    className: 'bg-blue-100 text-blue-800 border-blue-200',
  },
  pending: {
    variant: 'secondary' as const,
    className: 'bg-gray-100 text-gray-800 border-gray-200',
  },
  processing: {
    variant: 'secondary' as const,
    className: 'bg-orange-100 text-orange-800 border-orange-200',
  },
};

export function StatusBadge({ status, children, className }: StatusBadgeProps) {
  const config = statusConfig[status];

  return (
    <Badge
      variant={config.variant}
      className={cn(config.className, className)}
    >
      {children}
    </Badge>
  );
}