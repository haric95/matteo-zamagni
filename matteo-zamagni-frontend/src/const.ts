import { Plus, SelectableIconComponent, Rhombus } from "./components/Icons";
import { HomepageItemType } from "./types/global";

export const DEFAULT_ANIMATE_MODE = {
  initial: { opacity: 0 },
  exit: { opacity: 0 },
  animate: { opacity: 1 },
  transition: { type: "ease-in-out", duration: 0.5 },
};

export const homepageItemArray = [
  HomepageItemType.EXHIBITION,
  HomepageItemType.PROJECT,
];

export const HomepageItemTypeIconMap: Record<
  HomepageItemType,
  SelectableIconComponent
> = {
  [HomepageItemType.EXHIBITION]: Plus,
  [HomepageItemType.PROJECT]: Rhombus,
};
