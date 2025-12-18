"use client";

import { useSession, signOut } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Search,
  Bell,
  User,
  Settings,
  LogOut,
  HelpCircle,
  Command,
  Shield,
} from "lucide-react";

interface TopHeaderProps {
  className?: string;
}

interface DirectUser {
  id: string;
  email: string;
  name: string;
  role: string;
  department: string;
  permissions: string[];
}

export default function TopHeader({ className }: TopHeaderProps) {
  const { data: session } = useSession();
  const router = useRouter();
  const [directUser, setDirectUser] = useState<DirectUser | null>(null);
  const [authMethod, setAuthMethod] = useState<'nextauth' | 'direct' | null>(null);

  // Check for direct authentication
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const storedUser = localStorage.getItem('user');
        const storedAuthMethod = localStorage.getItem('authMethod');

        if (storedUser && storedAuthMethod === 'direct') {
          const user = JSON.parse(storedUser);
          setDirectUser(user);
          setAuthMethod('direct');
        } else if (session) {
          setAuthMethod('nextauth');
        }
      } catch (error) {
        console.error('Error reading direct auth from localStorage:', error);
      }
    }
  }, [session]);

  // Get current user info (either from session or direct auth)
  const currentUser = session?.user || directUser;
  const userInitials = currentUser?.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'U';

  const handleSignOut = async () => {
    if (authMethod === 'direct') {
      // Handle direct auth logout
      try {
        await fetch('/api/auth/direct-logout', {
          method: 'POST',
          headers: {
            'ngrok-skip-browser-warning': '69420'
          }
        });

        // Clear localStorage
        localStorage.removeItem('user');
        localStorage.removeItem('authMethod');

        // Redirect to home page
        router.push('/');
      } catch (error) {
        console.error('Direct logout error:', error);
        // Force logout by clearing storage and redirecting
        localStorage.removeItem('user');
        localStorage.removeItem('authMethod');
        router.push('/');
      }
    } else {
      // Handle NextAuth logout
      signOut({ callbackUrl: '/' });
    }
  };

  return (
    <header className={`bg-background/95 backdrop-blur-sm border-b transition-all duration-300 ${className}`}>
      <div className="flex items-center justify-between px-4 sm:px-6 py-3 sm:py-4 gap-2 sm:gap-4">
        {/* Search Bar - Responsive width and visibility */}
        <div className="flex-1 max-w-xs sm:max-w-md lg:max-w-xl">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground transition-colors" />
            <Input
              placeholder="Search..."
              className="pl-10 pr-4 sm:pr-14 text-sm transition-all duration-200 focus:ring-2 focus:ring-ring"
            />
            {/* Command+K shortcut - hidden on mobile */}
            <kbd className="hidden sm:inline-flex absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground opacity-100">
              <Command className="h-3 w-3" />K
            </kbd>
          </div>
        </div>

        {/* Right Section - Responsive spacing */}
        <div className="flex items-center gap-1 sm:gap-2 md:gap-4">
          {/* Notifications - Responsive sizing */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button 
                variant="ghost" 
                size="icon" 
                className="relative transition-all duration-200 hover:bg-accent h-9 w-9 sm:h-10 sm:w-10"
                aria-label="Notifications"
              >
                <Bell className="h-4 w-4 sm:h-5 sm:w-5 transition-all" />
                <Badge
                  variant="destructive"
                  className="absolute -top-1 -right-1 h-4 w-4 sm:h-5 sm:w-5 rounded-full p-0 flex items-center justify-center text-[10px] sm:text-xs transition-all"
                >
                  3
                </Badge>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-72 sm:w-80 md:w-96 transition-all">
              <DropdownMenuLabel>Notifications</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="flex flex-col items-start p-3 sm:p-4 transition-all duration-200 hover:bg-accent cursor-pointer">
                <div className="flex items-center space-x-2 w-full">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  <span className="font-medium text-sm">Clash Detection Complete</span>
                  <span className="text-xs text-muted-foreground ml-auto whitespace-nowrap">2m ago</span>
                </div>
                <p className="text-xs sm:text-sm text-muted-foreground mt-1">
                  Found 3 clashes in Project Alpha BIM model
                </p>
              </DropdownMenuItem>
              <DropdownMenuItem className="flex flex-col items-start p-3 sm:p-4 transition-all duration-200 hover:bg-accent cursor-pointer">
                <div className="flex items-center space-x-2 w-full">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="font-medium text-sm">Document Processed</span>
                  <span className="text-xs text-muted-foreground ml-auto whitespace-nowrap">5m ago</span>
                </div>
                <p className="text-xs sm:text-sm text-muted-foreground mt-1">
                  CAD drawings successfully converted to 3D
                </p>
              </DropdownMenuItem>
              <DropdownMenuItem className="flex flex-col items-start p-3 sm:p-4 transition-all duration-200 hover:bg-accent cursor-pointer">
                <div className="flex items-center space-x-2 w-full">
                  <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                  <span className="font-medium text-sm">PM Bot Update</span>
                  <span className="text-xs text-muted-foreground ml-auto whitespace-nowrap">10m ago</span>
                </div>
                <p className="text-xs sm:text-sm text-muted-foreground mt-1">
                  Task assignment completed for Foundation Phase
                </p>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Help - Hidden on mobile to save space */}
          <Button 
            variant="ghost" 
            size="icon"
            className="hidden sm:inline-flex transition-all duration-200 hover:bg-accent h-9 w-9 sm:h-10 sm:w-10"
            aria-label="Help"
          >
            <HelpCircle className="h-4 w-4 sm:h-5 sm:w-5" />
          </Button>

          {/* User Menu - Responsive sizing */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button 
                variant="ghost" 
                className="relative h-8 w-8 sm:h-9 sm:w-9 rounded-full transition-all duration-200 hover:ring-2 hover:ring-ring hover:ring-offset-2"
                aria-label="User menu"
              >
                <Avatar className="h-8 w-8 sm:h-9 sm:w-9 transition-all">
                  <AvatarImage src="/avatars/01.png" alt="User" />
                  <AvatarFallback className="text-xs sm:text-sm">
                    {userInitials}
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-52 sm:w-56 transition-all" align="end" forceMount>
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium leading-none">{currentUser?.name || 'User'}</p>
                  <p className="text-xs leading-none text-muted-foreground">
                    {currentUser?.email || 'user@example.com'}
                  </p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-xs text-muted-foreground">Role: {currentUser?.role}</p>
                  <p className="text-xs text-muted-foreground">Dept: {currentUser?.department}</p>
                  {authMethod && (
                    <div className="flex items-center gap-1 mt-1">
                      <Shield className="w-3 h-3" />
                      <p className="text-xs text-muted-foreground">
                        Auth: {authMethod === 'nextauth' ? 'NextAuth.js' : 'Direct'}
                      </p>
                    </div>
                  )}
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="transition-all duration-200 hover:bg-accent cursor-pointer">
                <User className="mr-2 h-4 w-4 transition-transform hover:scale-110" />
                <span>Profile</span>
              </DropdownMenuItem>
              <DropdownMenuItem className="transition-all duration-200 hover:bg-accent cursor-pointer">
                <Settings className="mr-2 h-4 w-4 transition-transform hover:rotate-90" />
                <span>Settings</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem 
                onClick={handleSignOut}
                className="transition-all duration-200 hover:bg-destructive/10 hover:text-destructive cursor-pointer"
              >
                <LogOut className="mr-2 h-4 w-4 transition-transform hover:translate-x-1" />
                <span>Log out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
}
