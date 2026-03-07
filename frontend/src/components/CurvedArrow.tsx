import { useRef } from "react";
import { motion, useScroll, useTransform } from "motion/react";

export function CurvedArrow({ flip = false }: { flip?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start 0.9", "end 0.55"],
  });
  const pathLength = useTransform(scrollYProgress, [0, 1], [0, 1]);
  const arrowOpacity = useTransform(scrollYProgress, [0.85, 1], [0, 1]);

  const d = flip
    ? "M 180 5 C 170 85, 100 130, 80 185"
    : "M 120 5 C 130 85, 200 130, 220 185";

  const arrowD = flip
    ? "M 90.42 173.89 L 80 185 L 79.14 169.79 Z"
    : "M 209.58 173.89 L 220 185 L 220.86 169.79 Z";

  return (
    <div ref={ref} className="flex justify-center py-2">
      <svg
        viewBox="0 0 300 200"
        className="w-40 h-auto"
        fill="none"
        overflow="visible"
      >
        <motion.path
          d={d}
          className="stroke-arrow"
          strokeWidth={2.5}
          strokeDasharray="6 4"
          strokeLinecap="round"
          style={{ pathLength }}
        />
        <motion.path
          d={arrowD}
          className="stroke-arrow"
          fill="var(--color-arrow)"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ opacity: arrowOpacity }}
        />
      </svg>
    </div>
  );
}
