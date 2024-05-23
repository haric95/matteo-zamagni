import { useEffect } from "react";

export const usePrefetchImages = (imageURLs: string[] | null) => {
  useEffect(() => {
    imageURLs?.forEach((url) => {
      const image = new Image();
      image.src = url;
    });
  }, [imageURLs]);
};
