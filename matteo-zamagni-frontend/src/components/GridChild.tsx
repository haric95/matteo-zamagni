// A div element that should be placed as the immediate child of another grid element.
// Will automatically create a grid with the same cell size as the parent grid.
// Grid positions are 0 indexed

import { HTMLAttributes, PropsWithChildren } from "react";

type GridChildProps = {
  x: number;
  y: number;
  height: number;
  width: number;
} & HTMLAttributes<HTMLDivElement>;

export const GridChild: React.FC<PropsWithChildren<GridChildProps>> = ({
  x,
  y,
  height,
  width,
  children,
  className,
  style,
  ...divProps
}) => {
  return (
    <div
      className={`grid ${className}`}
      style={{
        gridColumnStart: x + 1,
        gridColumnEnd: x + 1 + width,
        gridRowStart: y + 1,
        gridRowEnd: y + 1 + height,
        gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${height}, minmax(0, 1fr))`,
        ...style,
      }}
      {...divProps}
    >
      {children}
    </div>
  );
};
