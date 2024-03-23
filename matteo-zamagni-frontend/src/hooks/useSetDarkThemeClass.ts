import { useGlobalContext } from "@/state/GlobalStore";
import { useEffect } from "react";

export const useSetDarkThemeClass = () => {
  const { isDark } = useGlobalContext();

  useEffect(() => {
    const htmlElement = document.querySelector("html");
    if (htmlElement) {
      if (isDark) {
        htmlElement.classList.add("dark");
      } else {
        htmlElement.classList.remove("dark");
      }
    }
  }, [isDark]);
};
