'use client';

import { ReactNode } from 'react';
import { useSession, signOut } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  User,
  Settings,
  LogOut,
  HelpCircle,
} from 'lucide-react';

interface HeaderProps {
  /** Custom content to display in the header */
  content?: ReactNode;
  /** Whether to show user menu */
  showUserMenu?: boolean;
  /** Custom user menu items */
  userMenuItems?: Array<{
    label: string;
    icon?: ReactNode;
    onClick?: () => void;
    href?: string;
  }>;
}

export function Header({
  content,
  showUserMenu = true,
  userMenuItems
}: HeaderProps) {
  const { data: session } = useSession();
  const router = useRouter();

  const defaultUserMenuItems = [
    {
      label: 'Profile',
      icon: <User className="h-4 w-4" />,
      onClick: () => router.push('/profile'),
    },
    {
      label: 'Settings',
      icon: <Settings className="h-4 w-4" />,
      onClick: () => router.push('/settings'),
    },
    {
      label: 'Help',
      icon: <HelpCircle className="h-4 w-4" />,
      onClick: () => router.push('/help'),
    },
    {
      label: 'Sign Out',
      icon: <LogOut className="h-4 w-4" />,
      onClick: () => signOut(),
    },
  ];

  const menuItems = userMenuItems || defaultUserMenuItems;

  return (
    <header className="bg-background border-b px-4 py-3 flex items-center justify-between">
      {/* Left side - Custom content */}
      <div className="flex-1">
        {content}
      </div>

      {/* Right side - User menu */}
      {showUserMenu && session?.user && (
        <div className="flex items-center gap-4">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                <Avatar className="h-8 w-8">
                  <AvatarImage src={session.user.image || ''} alt={session.user.name || ''} />
                  <AvatarFallback>
                    {session.user.name?.charAt(0).toUpperCase() || 'U'}
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56" align="end" forceMount>
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium leading-none">
                    {session.user.name}
                  </p>
                  <p className="text-xs leading-none text-muted-foreground">
                    {session.user.email}
                  </p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              {menuItems.map((item, index) => (
                <DropdownMenuItem
                  key={index}
                  onClick={item.onClick}
                  className="cursor-pointer"
                >
                  {item.icon}
                  <span className="ml-2">{item.label}</span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      )}
    </header>
  );
}