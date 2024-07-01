import { useEffect, useState } from "react";
import { useStrapi } from "./useStrapi";
import { HomepageData } from "@/types/global";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";

export const useLoadInitial = () => {
  const homepageData = useStrapi<HomepageData, false>("/homepage", {
    "populate[items][populate][0]": "position",
    "populate[items][populate][1]": "image",
  });
  const { scrollerAvailableYears, hasLoaded } = useGlobalContext();
  const dispatch = useGlobalContextDispatch();

  useEffect(() => {
    if (homepageData && !scrollerAvailableYears) {
      const years = new Set(["0000"]);
      homepageData.data.attributes.items.forEach((item) => {
        years.add(String(item.year));
      });
      const sortedYears = Array.from(years).sort(
        (a, b) => Number(a) - Number(b)
      );
      if (dispatch) {
        dispatch({ type: "SET_SCROLLER_AVAILABLE_YEARS", years: sortedYears });
      }
    }
  }, [scrollerAvailableYears, homepageData, dispatch]);

  useEffect(() => {
    const dataIsLoaded = !!scrollerAvailableYears;

    if (!hasLoaded && dataIsLoaded && dispatch) {
      dispatch({ type: "SET_LOADED" });
    }
  }, [dispatch, hasLoaded, scrollerAvailableYears]);
};
