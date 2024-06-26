// A div element that should be placed as the immediate child of another grid element.
// Will automatically create a grid with the same cell size as the parent grid.
// Grid positions are 0 indexed

import { Dim2D, Pos2D } from "@/types/global";
import {
  HTMLAttributes,
  PropsWithChildren,
  RefAttributes,
  useMemo,
} from "react";

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
  ref?: RefAttributes<HTMLDivElement>["ref"];
} & HTMLAttributes<HTMLDivElement>;

export const getAbsGridCoords = (
  outerGridSize: Dim2D,
  propPos: Pos2D,
  mode: "floor" | "round" | "ceil" = "floor"
) => {
  // Subtract small value to make sure a propPos of 1 doesn't cause grid
  // coord values to not fit in grid
  const xCoord =
    propPos.x === 1
      ? outerGridSize.x
      : mode === "floor"
      ? Math.floor(outerGridSize.x * propPos.x)
      : mode === "ceil"
      ? Math.ceil(outerGridSize.x * propPos.x)
      : Math.round(outerGridSize.x * propPos.x);
  const yCoord =
    propPos.y === 1
      ? outerGridSize.y
      : mode === "floor"
      ? Math.floor(outerGridSize.y * propPos.y)
      : mode === "ceil"
      ? Math.ceil(outerGridSize.y * propPos.y)
      : Math.round(outerGridSize.y * propPos.y);

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
  ref,
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
        ref={ref || null}
        {...divProps}
      >
        {children}
      </div>
    )
  );
};
