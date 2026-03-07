import { useRef } from "react";
import { motion, useInView } from "motion/react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface TooltipItem {
  label: string;
  colour: string;
}

interface StepCardProps {
  num: string;
  image: string;
  caption: string;
  align: "left" | "right";
  tooltip?: TooltipItem[];
}

export function StepCard({ num, image, caption, align, tooltip }: StepCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true });

  const fromLeft = align === "left";

  return (
    <div
      ref={ref}
      className={cn("flex", fromLeft ? "justify-start" : "justify-end")}
    >
      <motion.div
        className="group relative"
        initial={{ opacity: 0, x: fromLeft ? -40 : 40 }}
        animate={isInView ? { opacity: 1, x: 0 } : {}}
        transition={{ duration: 0.5, ease: "easeOut" }}
      >
        <Card className="w-md overflow-hidden py-0 gap-0">
          <div>
            <img
              src={image}
              alt={caption}
              className="aspect-[7/5] w-full object-cover"
            />
          </div>
          <CardContent className="pt-3 pb-4">
            <motion.p
              className="font-display text-muted-foreground text-sm"
              initial={{ opacity: 0, y: 6 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.4, delay: 0.15, ease: "easeOut" }}
            >
              {num}
            </motion.p>
            <motion.p
              className="font-body text-lg mt-1"
              initial={{ opacity: 0, y: 6 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.4, delay: 0.3, ease: "easeOut" }}
            >
              {caption}
            </motion.p>
          </CardContent>
        </Card>
        {tooltip && tooltip.length > 0 && (
          <div className="pointer-events-none absolute left-full top-1/2 z-10 ml-4 -translate-y-1/2 rounded-lg border border-white/10 bg-black/75 px-3 py-2.5 opacity-0 backdrop-blur-sm transition-opacity duration-200 group-hover:opacity-100">
            <ul className="flex flex-col gap-1">
              {tooltip.map((item) => (
                <li key={item.label} className="flex items-center gap-2 text-sm text-white">
                  <span
                    className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full"
                    style={{ backgroundColor: item.colour }}
                  />
                  {item.label}
                </li>
              ))}
            </ul>
          </div>
        )}
      </motion.div>
    </div>
  );
}
