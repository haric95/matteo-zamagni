import { getStrapiURL } from "./api";

type StrapiMedia = {
  data: {
    attributes: {
      url: string;
    };
  };
};

export function getStrapiMedia(media: StrapiMedia) {
  const { url } = media.data.attributes;
  const imageUrl = url.startsWith("/") ? getStrapiURL(url) : url;
  return imageUrl;
}
