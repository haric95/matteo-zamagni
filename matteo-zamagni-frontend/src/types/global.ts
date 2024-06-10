import { StrapiImageResponse } from "@/hooks/useStrapi";

export type Dim2D = { x: number; y: number };
export type Pos2D = { x: number; y: number };
export type PolarPos2d = { radius: number; theta: number }; // Theta in radians
export type PosAndDim2D = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type Grid = boolean[][];

export enum HomepageItemType {
  EXHIBITION = "Exhibition",
  PROJECT = "Project",
}

export type HomepageItem = {
  position: { x: number; y: number };
  type: HomepageItemType;
  title: string;
  year: string;
  image: StrapiImageResponse;
  slug: string;
  tags: string;
};

export type HomepageData = {
  items: HomepageItem[];
};
