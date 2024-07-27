"use client";
import { AboutViewer } from "@/components/AboutViewer";
import { FooterRight } from "@/components/FooterRight";
import { GridChild } from "@/components/GridChild";
import { MotionGridChild } from "@/components/MotionGridChild";
import { DEFAULT_ANIMATE_MODE } from "@/const";
import { drawVerticalLine } from "@/helpers/gridHelpers";
import { useIsMobile } from "@/hooks/useIsMobile";
import { useLEDScrollbar } from "@/hooks/useLEDScrollbar";
import { useOnNavigate } from "@/hooks/useOnNavigate";
import { StrapiImageResponse, useStrapi } from "@/hooks/useStrapi";
import { useTheme } from "@/hooks/useTheme";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Dim2D, Grid, PosAndDim2D } from "@/types/global";
import { AnimatePresence, motion } from "framer-motion";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { TfiLayoutMenuV } from "react-icons/tfi";
import Markdown from "react-markdown";

const CENTER_CELL_PADDING_X = 16;
const CENTER_CELL_PADDING_Y = 6;
const CENTER_CELL_Y_OFFSET = 2;
const CENTER_CELL_PADDING_X_MOBILE = 2;
const CENTER_CELL_PADDING_Y_MOBILE = 6;
const CENTER_CELL_Y_OFFSET_MOBILE = 0;

enum AboutMode {
  BIO = "Bio",
  AWARDS = "Awards",
  RESIDENCIES = "Residencies",
  PERFORMANCES = "Performances",
  SCREENINGS = "Screenings",
  TALKS = "Talks",
}

export enum StrapiAboutComponentType {
  Title = "about.about-title",
  Year = "about.about-year",
  Item = "about.about-item",
  Text = "about.about-text",
}

type StrapiTitleComponent = {
  __component: StrapiAboutComponentType.Title;
  Title: string;
};

type StrapiYearComponent = {
  __component: StrapiAboutComponentType.Year;
  Year: string;
};

type StrapiItemComponent = {
  __component: StrapiAboutComponentType.Item;
  Label: string;
  Name: string;
  Details: string;
};

type StrapiTextComponent = {
  __component: StrapiAboutComponentType.Text;
  Text: string;
};

export type StrapiAboutComponent =
  | StrapiTitleComponent
  | StrapiYearComponent
  | StrapiItemComponent
  | StrapiTextComponent;

type AboutPageData = {
  [AboutMode.BIO]: StrapiAboutComponent[];
  [AboutMode.AWARDS]: StrapiAboutComponent[];
  [AboutMode.RESIDENCIES]: StrapiAboutComponent[];
  [AboutMode.PERFORMANCES]: StrapiAboutComponent[];
  [AboutMode.SCREENINGS]: StrapiAboutComponent[];
  [AboutMode.TALKS]: StrapiAboutComponent[];
  CV: StrapiImageResponse | null;
  DigitalSales: { label: string; url: string }[] | null;
  RepresentedBy: { label: string; url: string }[] | null;
};

// TODO: Add on mount delay to wait until bg color change has happened
// TODO: Add About Modes
export default function Home() {
  const aboutPageData = useStrapi<AboutPageData, false>("/about", {
    populate: "deep",
  });
  const { gridDim, grid } = useGlobalContext() as {
    gridDim: Dim2D;
    grid: Grid;
  };
  const isMobile = useIsMobile();
  const dispatch = useGlobalContextDispatch();

  const [aboutMode, setAboutMode] = useState<AboutMode>(AboutMode.BIO);
  const [ledIsSet, setLedIsSet] = useState(false);
  const [scrollDiv, setScrollDiv] = useState<HTMLDivElement | null>(null);

  const { shouldMount } = useTheme({ isDark: false });
  const centerCellPos = useMemo<PosAndDim2D>(() => {
    const width = isMobile
      ? gridDim.x - CENTER_CELL_PADDING_X_MOBILE * 2
      : gridDim.x - CENTER_CELL_PADDING_X * 2;
    const height = isMobile
      ? gridDim.y - CENTER_CELL_PADDING_Y_MOBILE * 2
      : gridDim.y - CENTER_CELL_PADDING_Y * 2;

    const yOffset = isMobile
      ? CENTER_CELL_Y_OFFSET_MOBILE
      : CENTER_CELL_Y_OFFSET;

    return {
      x: isMobile ? CENTER_CELL_PADDING_X_MOBILE : CENTER_CELL_PADDING_X,
      y:
        (isMobile ? CENTER_CELL_PADDING_Y_MOBILE : CENTER_CELL_PADDING_Y) +
        yOffset,
      width,
      height,
    };
  }, [gridDim, isMobile]);

  const CVCellPos = useMemo<PosAndDim2D>(() => {
    const width = isMobile ? 6 : 8;
    const height = isMobile ? 1 : 2;

    return {
      x: isMobile
        ? centerCellPos.x + centerCellPos.width - width + 1
        : centerCellPos.x + centerCellPos.width + 3,
      y: isMobile
        ? centerCellPos.y + centerCellPos.height + 1
        : centerCellPos.y + 2,
      width,
      height,
    };
  }, [isMobile, centerCellPos]);

  const infoCellPos = useMemo<PosAndDim2D>(() => {
    const width = isMobile ? 8 : 10;
    const height = isMobile ? 2 : 12;

    return {
      x: centerCellPos.x - width - 3,
      y: centerCellPos.y + centerCellPos.height / 2 - height / 2,
      width,
      height,
    };
  }, [isMobile, centerCellPos]);

  const handleScrollDivChange = useCallback((div: HTMLDivElement | null) => {
    if (div) {
      setScrollDiv(div);
    }
  }, []);

  useLEDScrollbar(
    centerCellPos.y,
    centerCellPos.y + centerCellPos.height - 1,
    centerCellPos.x + centerCellPos.width + 1,
    scrollDiv
  );

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
        centerCellPos.x - 1,
        centerCellPos.y,
        centerCellPos.y + centerCellPos.height
      ),
      centerCellPos.x + centerCellPos.width,
      centerCellPos.y,
      centerCellPos.y + centerCellPos.height
    );
    if (dispatch && !ledIsSet && shouldMount) {
      setLedIsSet(true);
      dispatch({ type: "CLEAR_GRID" });
      dispatch({ type: "UPDATE_GRID", grid: updatedGrid });
    }
  }, [centerCellPos, grid, dispatch, ledIsSet, shouldMount]);

  const handleNavigate = useCallback(() => {
    if (dispatch) {
      dispatch({ type: "CLEAR_GRID" });
    }
  }, [dispatch]);

  useOnNavigate(handleNavigate);

  return (
    <>
      <AnimatePresence>
        {shouldMount && (
          <>
            <MotionGridChild
              isGrid={false}
              {...CVCellPos}
              {...DEFAULT_ANIMATE_MODE}
              className="bg-background_Light"
              key={aboutMode}
            >
              <a
                className="w-full h-full flex justify-center items-center underline"
                href={
                  aboutPageData?.data?.attributes?.CV?.data?.attributes?.url ||
                  ""
                }
                download={"Matteo Zamagni CV"}
              >
                Download CV
              </a>
            </MotionGridChild>
          </>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {shouldMount && !isMobile && (
          <MotionGridChild
            isGrid={false}
            {...infoCellPos}
            {...DEFAULT_ANIMATE_MODE}
            className="bg-background_Light flex flex-col justify-around p-2"
            key={aboutMode}
          >
            <div>
              <p>represented by:</p>
              {aboutPageData?.data?.attributes?.RepresentedBy?.map((by) => (
                <Link
                  key={by.label}
                  href={by.url}
                  className="block underline"
                  target="_blank"
                >
                  {by.label}
                </Link>
              ))}
            </div>
            <div>
              <p>digital sales:</p>
              {aboutPageData?.data?.attributes?.DigitalSales?.map((by) => (
                <Link
                  key={by.label}
                  href={by.url}
                  className="block underline"
                  target="_blank"
                >
                  {by.label}
                </Link>
              ))}
            </div>
          </MotionGridChild>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {shouldMount && centerCellPos && (
          <MotionGridChild
            isGrid={false}
            {...centerCellPos}
            {...DEFAULT_ANIMATE_MODE}
            className="bg-background_Light"
            key={aboutMode}
          >
            <div
              ref={handleScrollDivChange}
              className="w-full h-full overflow-auto text-black whitespace-break-spaces no-scrollbar px-2 py-4"
            >
              <AboutViewer
                content={aboutPageData?.data?.attributes[aboutMode] || null}
              />
            </div>
          </MotionGridChild>
        )}
      </AnimatePresence>
      <FooterRight
        footerRightHeight={8}
        footerRightWidth={6}
        isMounted={shouldMount}
        mobileTitleComponent={
          <div className="flex items-center">
            <TfiLayoutMenuV color="white" className="mr-1" />
            <p>navigation - {aboutMode.toLocaleLowerCase()}</p>
          </div>
        }
      >
        <div
          className="grid col-span-full row-span-full  "
          style={{
            gridTemplateColumns: `repeat(${6}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${8}, minmax(0, 1fr))`,
          }}
        >
          <div
            className={`col-span-full row-span-1 flex items-start border-white md:border-black border-b-[1px] text-black`}
          >
            <p className="text-[12px] md:text-black text-white">navigation</p>
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
                  className={`text-[12px] block transition-color duration-500 ${
                    aboutMode === AboutMode.BIO
                      ? "text-white translate-x-2"
                      : "text-white md:text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.BIO);
                    if (dispatch) {
                      dispatch({
                        type: "SET_MOBILE_FOOTER_MENU",
                        isOpen: false,
                      });
                    }
                  }}
                >
                  bio
                </button>
                <button
                  className={`text-[12px] block transition-all duration-500 ${
                    aboutMode === AboutMode.AWARDS
                      ? "text-white translate-x-2"
                      : "text-white md:text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.AWARDS);
                    if (dispatch) {
                      dispatch({
                        type: "SET_MOBILE_FOOTER_MENU",
                        isOpen: false,
                      });
                    }
                  }}
                >
                  awards
                </button>
                <button
                  className={`text-[12px] block transition-all duration-500 ${
                    aboutMode === AboutMode.RESIDENCIES
                      ? "text-white translate-x-2"
                      : "text-white md:text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.RESIDENCIES);
                    if (dispatch) {
                      dispatch({
                        type: "SET_MOBILE_FOOTER_MENU",
                        isOpen: false,
                      });
                    }
                  }}
                >
                  residencies
                </button>
                <button
                  className={`text-[12px] block transition-all duration-500 ${
                    aboutMode === AboutMode.PERFORMANCES
                      ? "text-white translate-x-2"
                      : "text-white md:text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.PERFORMANCES);
                    if (dispatch) {
                      dispatch({
                        type: "SET_MOBILE_FOOTER_MENU",
                        isOpen: false,
                      });
                    }
                  }}
                >
                  performances
                </button>
                <button
                  className={`text-[12px] block transition-all duration-500 ${
                    aboutMode === AboutMode.SCREENINGS
                      ? "text-white translate-x-2"
                      : "text-white md:text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.SCREENINGS);
                    if (dispatch) {
                      dispatch({
                        type: "SET_MOBILE_FOOTER_MENU",
                        isOpen: false,
                      });
                    }
                  }}
                >
                  screenings
                </button>
                <button
                  className={`text-[12px] block transition-all duration-500 ${
                    aboutMode === AboutMode.TALKS
                      ? "text-white"
                      : "text-white md:text-black"
                  }`}
                  onClick={() => {
                    setAboutMode(AboutMode.TALKS);
                    if (dispatch) {
                      dispatch({
                        type: "SET_MOBILE_FOOTER_MENU",
                        isOpen: false,
                      });
                    }
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
