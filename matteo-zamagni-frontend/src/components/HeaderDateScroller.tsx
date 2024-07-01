//@ts-nocheck
// TSIGNORED due to out of date typings in keen-slider library

import { Cross } from "@/components/Icons";
import { DEFAULT_ANIMATE_MODE } from "@/const";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { AnimatePresence, motion } from "framer-motion";
import "keen-slider/keen-slider.min.css";
import { useKeenSlider } from "keen-slider/react";
import { usePathname } from "next/navigation";
import React, { useState } from "react";
import "./HeaderDateScroller.css";

const MULTIPLIER = 3;

// TODO: Resolve scroll issue that means visible overflow when reaching end of scroller wheel
export const HeaderDateScroller = (props) => {
  const dispatch = useGlobalContextDispatch();
  const { scrollerAvailableYears } = useGlobalContext();
  const CLONED_YEARS = scrollerAvailableYears
    ? new Array(MULTIPLIER).fill(scrollerAvailableYears).flat()
    : [];
  const pathname = usePathname();
  const perspective = "center";
  const wheelSize = CLONED_YEARS.length / MULTIPLIER;
  const slides = CLONED_YEARS.length;
  const slidesPerView = 1;
  const [sliderState, setSliderState] = useState(null);
  const [currentYearIndex, setCurrentYearIndex] = useState(0);
  const [sliderRef, slider] = useKeenSlider({
    perspective: "center",
    vertical: false,
    initial: 11,

    loop: true,
    dragSpeed: 1,
    move: (s) => {
      setSliderState(s.details());
    },
    rubberband: false,
    mode: "free-snap",
    slides,
    slidesPerView,
    afterChange: (s) => {
      // TODO: Try and implement some kind of automatic returning to the center of the wheel
      // so that the user is less likely to see the edges.
      // using moveToSlideRelative() does work but the wheel flickers when it makes the move
      // so need to find another way
      // const relativeSlide = s.details().relativeSlide;
      // if (relativeSlide < YEARS.length || relativeSlide >= YEARS.length * 2) {
      //   const index = (s.details().relativeSlide % YEARS.length) + YEARS.length;
      //   s.moveToSlideRelative(index, undefined, 0);
      // }
    },
    slideChanged: (s) => {
      if (scrollerAvailableYears) {
        const index = s.details().relativeSlide % scrollerAvailableYears.length;
        setCurrentYearIndex(
          s.details().relativeSlide % scrollerAvailableYears.length
        );
        dispatch({
          type: "SET_SELECTED_YEAR",
          year: index === 0 ? null : scrollerAvailableYears[index],
        });
      }
    },
  });

  const [radius, setRadius] = React.useState(0);

  React.useEffect(() => {
    if (slider) {
      setRadius(slider.details().widthOrHeight / 2);
    }
  }, [slider]);

  const slideValues = () => {
    if (!sliderState) {
      return [];
    }
    const offset = 0;

    return CLONED_YEARS.map((year, index) => {
      const distance = sliderState
        ? (sliderState.positions[index].distance - offset) * slidesPerView
        : 0;
      const rotate =
        Math.abs(distance) > wheelSize / 2
          ? 180
          : distance * (360 / wheelSize) * -1;
      const style = {
        transform: `rotateY(${rotate}deg) translateZ(${-radius}px)`,
        WebkitTransform: `rotateY(${rotate}deg) translateZ(${-radius}px)`,
        opacity: (1 - Math.abs(rotate) / 360) ** 6,
      };
      return { style, value: year };
    });
  };

  return (
    scrollerAvailableYears && (
      <div
        className={`w-full h-full relative ${
          pathname === "/" ? "" : "opacity-0 pointer-events-none"
        } transition-all duration-500 delay-500	`}
      >
        <div
          className={
            "absolute bg-background_Light dark:bg-background_Dark transition-all duration-500 font-decoration cursor-pointer wheel keen-slider wheel--perspective-" +
            perspective
          }
          ref={sliderRef}
        >
          <div className="wheel__inner">
            <div className="wheel__slides" style={{ width: 0 }}>
              {slideValues().map(({ style, value }, idx) => (
                <div className="wheel__slide" style={style} key={idx}>
                  <span
                    className="translate-x-1/2 dark:text-white text-black transition-all duration-500"
                    onClick={(e) => {
                      e.preventDefault();
                      slider.moveToSlide(idx, 1000);
                    }}
                  >
                    {value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="absolute top-0 w-full h-full flex justify-center pointer-events-none">
          <div className="w-24 border-x-[1px] dark:border-fadedWhite border-black transition-all duration-500"></div>
        </div>
        <AnimatePresence>
          {currentYearIndex !== 0 && (
            <motion.button
              {...DEFAULT_ANIMATE_MODE}
              className="absolute flex justify-center items-center top-0 w-4 h-4 top-[2px] right-[-32px]"
              onClick={() => {
                if (slider && scrollerAvailableYears) {
                  slider.moveToSlide(scrollerAvailableYears.length, 1000);
                }
              }}
            >
              <div className="w-[12px] h-[12px] interactable-button">
                <Cross
                  className="relative w-full h-full dark:stroke-white stroke-black"
                  strokeWidth={8}
                />
              </div>
            </motion.button>
          )}
        </AnimatePresence>
      </div>
    )
  );
};
