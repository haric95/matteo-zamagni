// A div element that should be placed as the immediate child of another grid element.
// Will automatically create a grid with the same cell size as the parent grid.
// Grid positions are 0 indexed

import { HTMLAttributes, PropsWithChildren } from "react";

type GridChildProps = {
  x: number;
  y: number;
  height: number;
  width: number;
  innerGridWidth?: number;
  innerGridHeight?: number;
  isGrid?: boolean;
} & HTMLAttributes<HTMLDivElement>;

export const GridChild: React.FC<PropsWithChildren<GridChildProps>> = ({
  x,
  y,
  height,
  width,
  children,
  className,
  innerGridWidth,
  innerGridHeight,
  isGrid = true,
  style,
  ...divProps
}) => {
  return (
    <div
      className={`${isGrid ? "grid" : ""} ${className ? className : ""}`}
      style={{
        gridColumnStart: x + 1,
        gridColumnEnd: x + 1 + width,
        gridRowStart: y + 1,
        gridRowEnd: y + 1 + height,
        gridTemplateColumns: `repeat(${
          innerGridWidth ? innerGridWidth : width
        }, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${
          innerGridHeight ? innerGridHeight : height
        }, minmax(0, 1fr))`,
        ...style,
      }}
      {...divProps}
    >
      {children}
    </div>
  );
};
