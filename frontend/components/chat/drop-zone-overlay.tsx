"use client";

import { AnimatePresence, motion } from "framer-motion";
import { FileTextIcon, ImageIcon } from "lucide-react";

export function DropZoneOverlay({ isVisible }: { isVisible: boolean }) {
  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          animate={{ opacity: 1, scale: 1 }}
          className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-foreground/30 bg-background/80 backdrop-blur-sm"
          exit={{ opacity: 0, scale: 0.98 }}
          initial={{ opacity: 0, scale: 0.98 }}
          transition={{ duration: 0.12 }}
        >
          <div className="flex items-center gap-2 text-muted-foreground">
            <ImageIcon className="size-5" />
            <FileTextIcon className="size-5" />
          </div>
          <p className="text-sm font-medium text-foreground">
            Drop image or PDF to attach
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
