import { useRef, useState } from "react"
import { Upload } from "lucide-react"
import { ACCEPT } from "@/lib/api"
import { cn } from "@/lib/utils"

export function UploadDropzone({ onFile, disabled }: { onFile: (f: File) => void; disabled?: boolean }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label="Upload a thermal image: PNG, JPG, JPEG or TIFF"
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !disabled && (e.preventDefault(), input.current?.click())}
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const f = e.dataTransfer.files[0]
        if (f && !disabled) onFile(f)
      }}
      className={cn(
        "flex cursor-pointer flex-col items-center gap-2 rounded-xl border border-dashed bg-white px-6 py-8 text-center transition-colors outline-none hover:border-primary/50 focus-visible:ring-3 focus-visible:ring-ring/50",
        over && "border-primary bg-primary/5",
        disabled && "pointer-events-none opacity-50",
      )}
    >
      <Upload className="size-5 text-muted-foreground" aria-hidden />
      <p className="text-sm font-medium">Drop a thermal image or click to browse</p>
      <p className="text-xs text-muted-foreground">PNG, JPG, JPEG or TIFF (8- or 16-bit) · max 20 MB</p>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onClick={(e) => e.stopPropagation()} // input lives inside the dropzone; don't re-trigger it
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
          e.target.value = "" // allow re-selecting the same file
        }}
      />
    </div>
  )
}
