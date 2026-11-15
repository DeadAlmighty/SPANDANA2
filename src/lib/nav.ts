import {
  Activity,
  BookOpen,
  Calculator,
  FileText,
  GitCompare,
  LayoutDashboard,
  UploadCloud,
  BrainCircuit,
  type LucideIcon,
} from "lucide-react";
import type { ViewId } from "@/store/analyses";

export interface NavItem {
  id: ViewId;
  label: string;
  icon: LucideIcon;
  hint?: string;
}

export const NAV_SECTIONS: { title: string; items: NavItem[] }[] = [
  {
    title: "Overview",
    items: [{ id: "dashboard", label: "Dashboard", icon: LayoutDashboard }],
  },
  {
    title: "ECG Workflow",
    items: [
      { id: "upload", label: "ECG Analysis", icon: UploadCloud, hint: "Upload" },
      { id: "visualization", label: "Signal Visualization", icon: Activity },
      { id: "criteria", label: "Traditional Analysis", icon: Calculator },
      { id: "screening", label: "AI Screening", icon: BrainCircuit },
    ],
  },
  {
    title: "Evaluation",
    items: [
      { id: "comparison", label: "Model Comparison", icon: GitCompare },
      { id: "reports", label: "Analysis Reports", icon: FileText },
    ],
  },
  {
    title: "Reference",
    items: [{ id: "research", label: "Research Information", icon: BookOpen }],
  },
];

/** Every valid module id, used for hash-route validation. */
export const ALL_VIEW_IDS: ViewId[] = NAV_SECTIONS.flatMap((s) =>
  s.items.map((i) => i.id),
);

export const VIEW_META: Record<
  ViewId,
  { title: string; subtitle: string; module: string }
> = {
  dashboard: {
    title: "Research Dashboard",
    subtitle: "Screening activity, cohort overview and model status at a glance.",
    module: "Module 1",
  },
  upload: {
    title: "ECG Analysis",
    subtitle: "Upload a 10-second, 12-lead ECG recording to run the screening pipeline.",
    module: "Module 2",
  },
  visualization: {
    title: "Signal Visualization",
    subtitle: "Inspect the 12-lead waveform with lead-by-lead labelling.",
    module: "Module 3",
  },
  criteria: {
    title: "Traditional Analysis",
    subtitle: "Sokolow–Lyon and Cornell voltage criteria computed from measured QRS amplitudes.",
    module: "Module 4",
  },
  screening: {
    title: "AI Screening",
    subtitle: "Perceptron and CNN LVH probability outputs with confidence breakdown.",
    module: "Module 5",
  },
  comparison: {
    title: "Model Comparison",
    subtitle: "Accuracy, precision, sensitivity, specificity, F1-score and ROC-AUC across methods.",
    module: "Module 6",
  },
  reports: {
    title: "Analysis Reports",
    subtitle: "Generate a professional research report for the selected recording.",
    module: "Module 7",
  },
  research: {
    title: "Research Information",
    subtitle: "LVH background, methodology, the PTB-XL dataset, model architecture and limitations.",
    module: "Module 8",
  },
};
