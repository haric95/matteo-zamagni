//@ts-nocheck
// TSIGNORED due to out of date typings in keen-slider library

import React, { useState } from "react";
import { useKeenSlider } from "keen-slider/react";
import { Cross } from "@/components/Icons";
import "keen-slider/keen-slider.min.css";
import "./HeaderDateScroller.css";

const YEARS = [
  "0000",
  "2015",
  "2016",
  "2017",
  "2018",
  "2019",
  "2020",
  "2021",
  "2022",
  "2023",
  "2024",
];

const MULTIPLIER = 3;

const CLONED_YEARS = new Array(MULTIPLIER).fill(YEARS).flat();

// TODO: Resolve scroll issue that means visible overflow when reaching end of scroller wheel
export const HeaderDateScroller = (props) => {
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
    slideChanged: (s) => {
      setCurrentYearIndex(s.details().relativeSlide % YEARS.length);
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
    <div className="w-full h-full relative">
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
                <span className="translate-x-1/2">{value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="absolute top-0 w-full h-full flex justify-center pointer-events-none">
        <div className="w-24 border-x-[1px] border-fadedWhite"></div>
      </div>
      {currentYearIndex !== 0 && (
        <button
          className="absolute flex justify-center items-center top-0 w-4 h-4 top-[2px] right-[-32px]"
          onClick={() => {
            if (slider) {
              slider.moveToSlide(YEARS.length, 1000);
            }
          }}
        >
          <div className="w-[12px] h-[12px]">
            <Cross
              className="relative w-full h-full"
              stroke="white"
              strokeWidth={8}
            />
          </div>
        </button>
      )}
    </div>
  );
};
