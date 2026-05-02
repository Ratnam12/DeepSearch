"use client";

import { UserButton, useUser } from "@clerk/nextjs";
import { useTheme } from "next-themes";
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { LoaderIcon } from "./icons";

export function SidebarUserNav() {
  const { isLoaded, user } = useUser();
  const { setTheme, resolvedTheme } = useTheme();

  if (!isLoaded) {
    return (
      <SidebarMenu>
        <SidebarMenuItem>
          <SidebarMenuButton className="h-10 justify-between rounded-lg bg-transparent text-sidebar-foreground/50">
            <div className="flex flex-row items-center gap-2">
              <div className="size-6 animate-pulse rounded-full bg-sidebar-foreground/10" />
              <span className="animate-pulse rounded-md bg-sidebar-foreground/10 text-transparent text-[13px]">
                Loading...
              </span>
            </div>
            <div className="animate-spin text-sidebar-foreground/50">
              <LoaderIcon />
            </div>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    );
  }

  if (!user) {
    return null;
  }

  const primaryEmail = user.primaryEmailAddress?.emailAddress ?? "";
  const display = user.fullName ?? primaryEmail;

  return (
    <SidebarMenu>
      <SidebarMenuItem className="flex items-center gap-2 px-2 py-1">
        <UserButton
          appearance={{
            elements: {
              avatarBox: "size-7 ring-1 ring-sidebar-border/50",
            },
          }}
          showName={false}
        />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] text-sidebar-foreground/80">
            {display}
          </div>
        </div>
        <button
          aria-label="Toggle theme"
          className="cursor-pointer text-[12px] text-sidebar-foreground/60 hover:text-sidebar-foreground"
          onClick={() =>
            setTheme(resolvedTheme === "dark" ? "light" : "dark")
          }
          type="button"
        >
          {resolvedTheme === "dark" ? "☀" : "☾"}
        </button>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
