// A div element that should be placed as the immediate child of another grid element.
// Will automatically create a grid with the same cell size as the parent grid.
// Grid positions are 0 indexed

import { Dim2D, Pos2D } from "@/types/global";
import { HTMLAttributes, PropsWithChildren, useMemo } from "react";

type GridChildProps = {
  x: number;
  y: number;
  posType?: "abs" | "prop"; // abs positions using grid coordinates, prop gives a proportion
  outerGridSize?: { height: number; width: number };
  height: number;
  width: number;
  innerGridWidth?: number;
  innerGridHeight?: number;
  isGrid?: boolean;
} & HTMLAttributes<HTMLDivElement>;

export const getAbsGridCoords = (outerGridSize: Dim2D, propPos: Pos2D) => {
  const xCoord = Math.floor(outerGridSize.x * propPos.x) + 1;
  const yCoord = Math.floor(outerGridSize.y * propPos.y) + 1;

  return { x: xCoord, y: yCoord };
};

export const GridChild: React.FC<PropsWithChildren<GridChildProps>> = ({
  x,
  y,
  posType = "abs",
  outerGridSize,
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
  const position = useMemo(() => {
    if (posType === "abs") {
      return {
        gridColumnStart: x + 1,
        gridColumnEnd: x + 1 + width,
        gridRowStart: y + 1,
        gridRowEnd: y + 1 + height,
      };
    } else if (outerGridSize && posType === "prop") {
      // If we are using proportion to position
      const absGridCoords = getAbsGridCoords(
        { x: outerGridSize.width, y: outerGridSize.height },
        { x: x, y: y }
      );

      return {
        gridColumnStart: absGridCoords.x,
        gridRowStart: absGridCoords.y,
        gridColumnEnd: absGridCoords.x + width,
        gridRowEnd: absGridCoords.y + height,
      };
    }
    return null;
  }, [posType, height, width, x, y, outerGridSize]);

  return (
    position && (
      <div
        className={`${isGrid ? "grid" : ""} ${className ? className : ""}`}
        style={{
          ...position,
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
    )
  );
};
