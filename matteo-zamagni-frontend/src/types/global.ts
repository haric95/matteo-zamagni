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
