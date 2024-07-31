export const getImageAspectRatio = async (
  imageUrl: string
): Promise<number | null> => {
  return new Promise((resolve, reject) => {
    const image = document.createElement("img");
    image.onload = () => {
      const aspect = image.height / image.width;
      resolve(aspect);
      image.remove();
    };
    image.onerror = () => reject();
    image.src = imageUrl;
  });
};
