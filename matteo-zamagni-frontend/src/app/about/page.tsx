"use client";
import { FooterRight } from "@/components/FooterRight";
import { drawVerticalLine } from "@/helpers/gridHelpers";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D, Grid } from "@/types/global";
import { useEffect, useMemo, useState } from "react";

const CENTER_CELL_WIDTH_PROPOPRTION = 0.4;
const CENTER_CELL_HEIGHT_PROPORTION = 0.5;
const CENTER_CELL_OFFSET_PROPORTION = 0.05;

enum AboutMode {
  BIO,
  AWARDS,
  RESIDENCIES,
  PERFORMANCES,
  SCREENINGS,
  TALKS,
}

// TODO: Add on mount delay to wait until bg color change has happened
// TODO: Add About Modes
export default function Home() {
  const { gridDim, grid } = useGlobalContext() as {
    gridDim: Dim2D;
    grid: Grid;
  };
  const dispatch = useGlobalContextDispatch();

  const [aboutMode, setAboutMode] = useState<AboutMode>(AboutMode.BIO);
  const [ledIsSet, setLedIsSet] = useState(false);

  const centerCellPos = useMemo(() => {
    const width =
      Math.floor(gridDim.x * 0.5 * CENTER_CELL_WIDTH_PROPOPRTION) * 2;
    const height =
      Math.floor(gridDim.y * 0.5 * CENTER_CELL_HEIGHT_PROPORTION) * 2;

    const yCenterOffest = Math.floor(gridDim.y * CENTER_CELL_OFFSET_PROPORTION);

    return {
      colStart: 1 + gridDim.x / 2 - width / 2,
      colEnd: 1 + gridDim.x / 2 + width / 2,
      rowStart: 1 + gridDim.y / 2 + yCenterOffest - height / 2,
      rowEnd: 1 + gridDim.y / 2 + yCenterOffest + height / 2,
    };
  }, [gridDim]);

  useEffect(() => {
    if (gridDim) {
      setLedIsSet(false);
    }
  }, [gridDim]);

  useEffect(() => {
    const updatedGrid = drawVerticalLine(
      drawVerticalLine(
        grid,
        // grid is 0 indexed and we want to highlight the column on the outside of the box
        centerCellPos.colStart - 2,
        centerCellPos.rowStart - 1,
        centerCellPos.rowEnd - 1
      ),
      centerCellPos.colEnd - 1,
      centerCellPos.rowStart - 1,
      centerCellPos.rowEnd - 1
    );
    if (dispatch && !ledIsSet) {
      setLedIsSet(true);
      dispatch({ type: "CLEAR_GRID" });
      dispatch({ type: "UPDATE_GRID", grid: updatedGrid });
    }
  }, [centerCellPos, grid, dispatch, ledIsSet]);

  useEffect(() => {
    if (dispatch) {
      dispatch({ type: "SET_IS_DARK", val: false });
    }
  }, [dispatch]);

  return (
    <>
      <div
        className="bg-background_Light"
        style={{
          gridColumnStart: centerCellPos.colStart,
          gridColumnEnd: centerCellPos.colEnd,
          gridRowStart: centerCellPos.rowStart,
          gridRowEnd: centerCellPos.rowEnd,
          // gridTemplateColumns: `repeat(${SIDE_HEADER_CELL_WIDTH}, minmax(0, 1fr))`,
          // gridTemplateRows: `repeat(${HEADER_UPPER_HEIGHT}, minmax(0, 1fr))`,
        }}
      >
        <div className="w-full h-full overflow-auto">
          <p>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi
            finibus neque nulla, vel tincidunt risus dignissim non. Integer
            euismod nisl ligula, non lobortis felis sagittis non. Fusce accumsan
            vestibulum metus vitae semper. Etiam convallis viverra augue vitae
            tempus. Suspendisse sodales, dui in molestie semper, sem nisl dictum
            mi, id congue lacus metus euismod nisi. Donec non ipsum nibh. In hac
            habitasse platea dictumst. Aliquam erat volutpat. Aenean ante
            mauris, pretium ac est vel, varius malesuada sem. Proin volutpat
            porttitor lectus. Etiam dignissim mi id diam sollicitudin vulputate.
            Aliquam auctor nulla at lacus scelerisque interdum quis eget elit.
            Fusce finibus arcu sed maximus posuere. Etiam magna velit, molestie
            in imperdiet ac, efficitur non nibh.{" "}
          </p>
          <br />
          <p>
            In hac habitasse platea dictumst. Duis quis tortor consectetur,
            pulvinar nisl et, tincidunt est. Curabitur ac eros et ligula maximus
            mollis a sit amet dolor. Praesent tempor vulputate felis, sit amet
            iaculis lorem bibendum id. Vivamus quam diam, volutpat quis purus
            quis, tincidunt hendrerit ex. Aliquam est metus, mollis vitae
            dignissim ac, varius ac metus. Quisque porttitor orci mi, vel
            efficitur tellus porttitor ut. Sed tincidunt est in tortor pulvinar
            porta. Donec aliquet elit sed nunc ornare, eget feugiat odio varius.
            In velit nulla, scelerisque ornare sem vel, scelerisque porttitor
            felis. Praesent consequat augue at dapibus dapibus. Ut et
            scelerisque elit. Morbi facilisis id turpis non varius. Fusce lorem
            velit, congue in dolor sit amet, malesuada venenatis magna.
            Pellentesque habitant morbi tristique senectus et netus et malesuada
            fames ac turpis egestas. Mauris ullamcorper laoreet lobortis. Nulla
            id turpis ut leo varius pulvinar.
          </p>
        </div>
      </div>
      <FooterRight footerRightHeight={8} footerRightWidth={6}>
        <div
          className="grid col-span-full row-span-full  "
          style={{
            gridTemplateColumns: `repeat(${6}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${8}, minmax(0, 1fr))`,
          }}
        >
          <div className="col-span-full row-span-1 flex items-start border-black border-b-[1px]">
            <p className="text-[12px]">navigation</p>
          </div>
          <div
            className={`col-span-full flex items-start`}
            style={{
              gridRowStart: 2,
              gridRowEnd: 100,
            }}
          >
            <div className="w-full h-full flex flex-col justify-center items-start">
              <div className="w-1/2 h-full flex flex-col justify-around items-start py-2">
                <button
                  className={`text-[12px] block ${
                    aboutMode === AboutMode.BIO ? "text-black" : "text-white"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.BIO);
                  }}
                >
                  bio
                </button>
                <button
                  className={`text-[12px] block ${
                    aboutMode === AboutMode.AWARDS ? "text-black" : "text-white"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.AWARDS);
                  }}
                >
                  awards
                </button>
                <button
                  className={`text-[12px] block ${
                    aboutMode === AboutMode.RESIDENCIES
                      ? "text-black"
                      : "text-white"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.RESIDENCIES);
                  }}
                >
                  residencies
                </button>
                <button
                  className={`text-[12px] block ${
                    aboutMode === AboutMode.PERFORMANCES
                      ? "text-black"
                      : "text-white"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.PERFORMANCES);
                  }}
                >
                  performances
                </button>
                <button
                  className={`text-[12px] block ${
                    aboutMode === AboutMode.SCREENINGS
                      ? "text-black"
                      : "text-white"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.SCREENINGS);
                  }}
                >
                  screenings
                </button>
                <button
                  className={`text-[12px] block ${
                    aboutMode === AboutMode.TALKS ? "text-black" : "text-white"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.TALKS);
                  }}
                >
                  talks
                </button>
              </div>
            </div>
          </div>
        </div>
      </FooterRight>
    </>
  );
}
