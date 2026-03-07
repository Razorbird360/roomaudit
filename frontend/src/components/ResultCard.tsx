import { motion } from "motion/react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export interface Defect {
  object: string;
  type: string;
  description: string;
}

export interface InspectResult {
  clean: boolean;
  defects: Defect[];
  crop_regions?: [number, number, number, number][];
  crop_reasons?: (string | null | undefined)[];
}

function formatSnakeCase(s: string): string {
  return s
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export function ResultCard({ result }: { result: InspectResult }) {
  return (
    <motion.div
      className="w-full"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            Inspection Result
            {result.clean ? (
              <Badge className="bg-pass text-white border-pass">Pass</Badge>
            ) : (
              <Badge className="bg-fail text-white border-fail">Fail</Badge>
            )}
          </CardTitle>
        </CardHeader>

        {!result.clean && result.defects.length > 0 && (
          <CardContent>
            <ul className="space-y-4">
              {result.defects.map((defect, i) => (
                <li key={i} className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-body font-medium">
                      {formatSnakeCase(defect.object)}
                    </span>
                    <Badge variant="secondary">{defect.type}</Badge>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    {defect.description}
                  </p>
                </li>
              ))}
            </ul>
          </CardContent>
        )}
      </Card>
    </motion.div>
  );
}
