"use client";
import { TARGET_CELL_SIZE } from "@/hooks/useScreenDim";
import { useStrapi } from "@/hooks/useStrapi";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { IndexPageData } from "@/types/global";
import Image from "next/image";
import React, {
  PropsWithChildren,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

const GRID_PIXEL_SIZE = 1.5; // LED pixel size

const PIXEL_TRANSITION_DURATION = 500;

export const PixelGrid: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim, grid } = useGlobalContext();
  const dispatch = useGlobalContextDispatch();
  const [cell, setCell] = useState<HTMLDivElement | null>(null);
  const aboutPageData = useStrapi<IndexPageData, false>("/homepage", {
    populate: "deep",
  });

  const handleThemeTransitionEnd = useCallback(() => {
    if (dispatch) {
      dispatch({
        type: "END_THEME_TRANSITION",
      });
    }
  }, [dispatch]);

  const cellRef = useRef<HTMLDivElement | null>(null);

  const updateCellSize = useCallback(() => {
    if (cellRef.current && dispatch) {
      const rect = cellRef.current.getBoundingClientRect();
      dispatch({
        type: "SET_CELL_SIZE",
        dim: {
          x: rect.width,
          y: rect.height,
        },
      });
    }
  }, [cellRef, dispatch]);

  useEffect(() => {
    // TODO(HC): Find a way to not use timeout here.
    // This is a hack for now, as the loading screen will cover the layout effects of this timeout.
    setTimeout(() => {
      if (cellRef) {
        updateCellSize();
      }
    }, 100);
  }, [cellRef, updateCellSize]);

  useEffect(() => {
    window.addEventListener("resize", updateCellSize);

    return () => {
      window.removeEventListener("resize", updateCellSize);
    };
  }, [updateCellSize]);

  return (
    <main className="fixed relative w-screen h-[calc(100dvh)] flex justify-center items-center bg-white overflow-hidden">
      <div
        className="absolute w-full h-full bg-background_Light dark:bg-background_Dark transition-[background-color] duration-500"
        onTransitionEnd={handleThemeTransitionEnd}
      />
      <div
        className="absolute bg-ledInactive_Light dark:bg-ledInactive_Dark"
        style={{
          width: gridDim ? `${gridDim.x * TARGET_CELL_SIZE}px` : "100%",
          height: gridDim ? `${gridDim.y * TARGET_CELL_SIZE}px` : "100%",
        }}
      />
      <div
        className="absolute"
        style={{
          width: gridDim ? `${gridDim.x * TARGET_CELL_SIZE}px` : "100%",
          height: gridDim ? `${gridDim.y * TARGET_CELL_SIZE}px` : "100%",
        }}
      >
        {aboutPageData && (
          <Image
            src={
              aboutPageData?.data?.attributes?.PixelBackgroundAnimation?.data
                ?.attributes?.url
            }
            className="bg-ledActive_Light dark:bg-ledActive_Dark opacity-0 dark:opacity-25 transition-opacity duration-1000"
            alt="background animation"
            objectFit="cover"
            layout="fill"
          />
        )}
      </div>
      <div
        className={`absolute grid pointer-events-none`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
          width: gridDim ? `${gridDim.x * TARGET_CELL_SIZE}px` : "100%",
          height: gridDim ? `${gridDim.y * TARGET_CELL_SIZE}px` : "100%",
        }}
      >
        <div
          className={`flex justify-center items-center transition-all border-background_Light dark:border-background_Dark`}
          key={`0-0`}
          style={{
            borderWidth: grid && grid[0][0] ? "9px" : "9.5px",
            transitionDuration: `${PIXEL_TRANSITION_DURATION}ms`,
          }}
          ref={cellRef}
        >
          {grid && grid[0][0] && (
            <div
              key={`0-0`}
              className="w-[2px] h-[2px] bg-ledActive_Light dark:bg-ledActive_Dark fade-in"
            />
          )}
        </div>
        {grid &&
          grid.map((columns, rowIndex) => {
            return columns.map((pixelIsLit, columnIndex) => {
              if (rowIndex === 0 && columnIndex === 0) {
                return;
              }
              return (
                <div
                  className={`flex justify-center items-center transition-all border-background_Light dark:border-background_Dark`}
                  key={`${rowIndex}-${columnIndex}`}
                  style={{
                    borderWidth: pixelIsLit ? "9px" : "9.5px",
                    transitionDuration: `${PIXEL_TRANSITION_DURATION}ms`,
                  }}
                >
                  {/* {pixelIsLit && (
                    <div
                      key={`${rowIndex}=${columnIndex}`}
                      className="w-[2px] h-[2px] bg-ledActive_Light dark:bg-ledActive_Dark fade-in"
                    />
                  )} */}
                </div>
              );
            });
          })}
      </div>
      <div
        className={`absolute grid`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
          width: gridDim ? `${gridDim.x * TARGET_CELL_SIZE}px` : "100%",
          height: gridDim ? `${gridDim.y * TARGET_CELL_SIZE}px` : "100%",
        }}
      >
        {children}
      </div>
    </main>
  );
};
