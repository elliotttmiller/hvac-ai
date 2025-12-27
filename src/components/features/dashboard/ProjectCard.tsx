'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { 
  Building2, 
  MapPin, 
  Calendar,
  Clock,
  Users,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProjectCardProps {
  id: string;
  name: string;
  location?: string;
  status?: string;
  progress?: number;
  createdAt?: string;
  teamMembers?: Array<{
    name: string;
    avatar?: string;
  }>;
  components?: number;
  documentsCount?: number;
  className?: string;
}

/**
 * ProjectCard - Snetch-inspired dark mode project card
 * Features: Hover effects, progress bars, avatar stacks, status badges
 */
export function ProjectCard({
  id,
  name,
  location = 'Unknown Location',
  status = 'active',
  progress = 0,
  createdAt,
  teamMembers = [],
  components = 0,
  documentsCount = 0,
  className,
}: ProjectCardProps) {
  const router = useRouter();

  const handleClick = () => {
    router.push(`/workspace/${id}`);
  };

  // Status badge variant mapping
  const statusVariant = status === 'active' ? 'default' : 
                       status === 'completed' ? 'secondary' : 
                       'outline';

  const statusColor = status === 'active' ? 'text-emerald-500' : 
                     status === 'completed' ? 'text-blue-500' : 
                     'text-slate-500';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2 }}
    >
      <Card
        onClick={handleClick}
        className={cn(
          'bg-slate-800 border-slate-700 hover:bg-slate-700/50 transition-all duration-200 cursor-pointer',
          'hover:shadow-lg hover:shadow-slate-900/50',
          className
        )}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3 flex-1">
              <div className="p-2 rounded-lg bg-slate-900/50 border border-slate-700">
                <Building2 className="w-5 h-5 text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <CardTitle className="text-base font-semibold text-white mb-1 truncate">
                  {name}
                </CardTitle>
                <div className="flex items-center gap-1 text-xs text-slate-400">
                  <MapPin className="w-3 h-3" />
                  <span className="truncate">{location}</span>
                </div>
              </div>
            </div>
            <Badge 
              variant={statusVariant}
              className={cn('text-xs capitalize', statusColor)}
            >
              {status}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Progress Bar */}
          {progress > 0 && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Progress</span>
                <span className="text-slate-300 font-medium">{progress}%</span>
              </div>
              <Progress value={progress} className="h-1.5 bg-slate-900" />
            </div>
          )}

          {/* Stats */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <div className="text-xs text-slate-400">Documents</div>
              <div className="text-lg font-bold text-white">{documentsCount}</div>
            </div>
            <div className="space-y-1">
              <div className="text-xs text-slate-400">Components</div>
              <div className="text-lg font-bold text-white">{components}</div>
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between pt-2 border-t border-slate-700">
            {/* Team Members Avatar Stack */}
            {teamMembers.length > 0 ? (
              <div className="flex -space-x-2">
                {teamMembers.slice(0, 3).map((member, idx) => (
                  <Avatar key={idx} className="w-6 h-6 border-2 border-slate-800">
                    <AvatarImage src={member.avatar} alt={member.name} />
                    <AvatarFallback className="text-xs bg-slate-700 text-slate-300">
                      {member.name.charAt(0)}
                    </AvatarFallback>
                  </Avatar>
                ))}
                {teamMembers.length > 3 && (
                  <div className="w-6 h-6 rounded-full bg-slate-700 border-2 border-slate-800 flex items-center justify-center">
                    <span className="text-[10px] text-slate-300">+{teamMembers.length - 3}</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-1 text-xs text-slate-500">
                <Users className="w-3 h-3" />
                <span>No team</span>
              </div>
            )}

            {/* Date */}
            {createdAt && (
              <div className="flex items-center gap-1 text-xs text-slate-400">
                <Clock className="w-3 h-3" />
                <span>{new Date(createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
