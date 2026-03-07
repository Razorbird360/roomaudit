import { StepCard } from "@/components/StepCard";
import { CurvedArrow } from "@/components/CurvedArrow";
import HIW1 from "@/assets/HIW1.jpg";
import HIW2 from "@/assets/HIW2.jpg";
import HIW3 from "@/assets/HIW3.jpg";
import HIW4 from "@/assets/HIW4.jpg";

// Detected objects from SAM3 on HIW2 — hardcoded from gen_howto_step1.py output
const hiw2Tooltip = [
  { label: "pillow",  colour: "rgb(200, 30, 30)"  },
  { label: "bed",     colour: "rgb(210, 110, 0)"  },
  { label: "floor",   colour: "rgb(20, 140, 20)"  },
  { label: "chair",   colour: "rgb(0, 110, 210)"  },
  { label: "window",  colour: "rgb(100, 100, 100)" },
];

const steps = [
  {
    num: "01",
    image: HIW1,
    caption: "We start with clean hotel room images",
    align: "left" as const,
  },
  {
    num: "02",
    image: HIW2,
    caption: "SAM3 detects and segments objects in the room",
    align: "right" as const,
    tooltip: hiw2Tooltip,
  },
  {
    num: "03",
    image: HIW3,
    caption: "FLUX.1 Fill inpaints realistic defects onto detected regions",
    align: "left" as const,
  },
  {
    num: "04",
    image: HIW4,
    caption: "Qwen3-VL inspects the room and returns structured results",
    align: "right" as const,
  },
];

export function HowItWorks() {
  return (
    <div className="flex flex-col py-16 gap-8">
      {steps.map((step, i) => (
        <div key={step.num}>
          <StepCard
            num={step.num}
            image={step.image}
            caption={step.caption}
            align={step.align}
            tooltip={step.tooltip}
          />
          {i < steps.length - 1 && (
            <CurvedArrow flip={step.align === "right"} />
          )}
        </div>
      ))}
    </div>
  );
}
