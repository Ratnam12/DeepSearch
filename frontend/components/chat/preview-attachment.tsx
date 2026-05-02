import { FileTextIcon } from "lucide-react";
import Image from "next/image";
import type { Attachment } from "@/lib/types";
import { Spinner } from "../ui/spinner";
import { CrossSmallIcon } from "./icons";

export const PreviewAttachment = ({
  attachment,
  isUploading = false,
  onRemove,
}: {
  attachment: Attachment;
  isUploading?: boolean;
  onRemove?: () => void;
}) => {
  const { name, url, contentType, pageCount } = attachment;
  const isImage = contentType?.startsWith("image");
  const isPdf = contentType === "application/pdf";

  return (
    <div
      className="group relative h-24 w-24 shrink-0 overflow-hidden rounded-xl border border-border/40 bg-muted"
      data-testid="input-attachment-preview"
    >
      {isImage ? (
        <Image
          alt={name ?? "attachment"}
          className="size-full object-cover"
          height={96}
          src={url}
          width={96}
        />
      ) : isPdf ? (
        <a
          className="flex size-full flex-col items-center justify-center gap-1 px-2 text-center text-muted-foreground transition-colors hover:text-foreground"
          href={url}
          rel="noreferrer"
          target="_blank"
          title={name}
        >
          <FileTextIcon className="size-6 shrink-0" />
          <span className="line-clamp-2 break-all text-[10px] leading-tight">
            {name}
          </span>
          {typeof pageCount === "number" && (
            <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[9px] tabular-nums text-muted-foreground">
              {pageCount} {pageCount === 1 ? "page" : "pages"}
            </span>
          )}
        </a>
      ) : (
        <div className="flex size-full items-center justify-center text-muted-foreground text-xs">
          File
        </div>
      )}

      {isUploading && (
        <div
          className="absolute inset-0 flex items-center justify-center rounded-xl bg-black/40 backdrop-blur-sm"
          data-testid="input-attachment-loader"
        >
          <Spinner className="size-5" />
        </div>
      )}

      {onRemove && !isUploading && (
        <button
          className="absolute top-1.5 right-1.5 flex size-5 items-center justify-center rounded-full bg-black/60 text-white opacity-0 backdrop-blur-sm transition-opacity hover:bg-black/80 group-hover:opacity-100"
          onClick={onRemove}
          type="button"
        >
          <CrossSmallIcon size={10} />
        </button>
      )}
    </div>
  );
};
