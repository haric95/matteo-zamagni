// Import Swiper React components
import Image from "next/image";
import { MdChevronLeft, MdChevronRight, MdClose } from "react-icons/md";
import { Swiper as SwiperCore } from "swiper/core";
import { Pagination } from "swiper/modules";
import { Swiper, SwiperSlide } from "swiper/react";

// Import Swiper styles
import { useEffect, useState } from "react";
import "swiper/css";
import "swiper/css/pagination";

type ImageGalleryProps = {
  images: { imageURL: string; alt: string }[];
  initialSlide: number;
  handleClose: () => void;
};

export const ImageGallery: React.FC<ImageGalleryProps> = ({
  images,
  initialSlide,
  handleClose,
}) => {
  const [swiper, setSwiper] = useState<SwiperCore | null>(null);
  const [currentSlide, setCurrentSlide] = useState(initialSlide);

  const handleClick = (dir: "left" | "right") => {
    if (swiper) {
      if (dir === "left") {
        setCurrentSlide((old) => (old + images.length - 1) % images.length);
      } else {
        setCurrentSlide((old) => (old + 1) % images.length);
      }
    }
  };

  useEffect(() => {
    if (swiper) {
      swiper.slideTo(currentSlide);
    }
  }, [swiper, currentSlide]);

  return (
    <div className="w-screen h-screen fixed left-0 top-0 z-10 flex justify-center items-center">
      <div
        className="w-full h-full absolute bg-black opacity-90"
        onClick={handleClose}
      />
      <div className="w-3/4 h-3/4 flex justify-center items-center z-10 relative">
        <button
          className="absolute top-0 right-0 w-8 h-8 flex justify-center"
          onClick={() => handleClose()}
        >
          <MdClose />
        </button>
        <button
          className="w-[5%] h-1/2 flex items-center justify-center"
          onClick={() => handleClick("left")}
        >
          <div className="h-fit w-fit">
            <MdChevronLeft width={16} height={16} />
          </div>
        </button>
        <Swiper
          // install Swiper modules
          modules={[Pagination]}
          onSwiper={(swiper) => setSwiper(swiper)}
          // modules={[Pagination, Navigation]}
          spaceBetween={0}
          slidesPerView={1}
          initialSlide={initialSlide}
          navigation
          pagination={{ clickable: true }}
          className="w-full h-full"
          style={{
            // @ts-ignore
            "--swiper-pagination-color": "#FEF781",
            "--swiper-pagination-bullet-inactive-color": "#999999",
            "--swiper-pagination-bullet-inactive-opacity": "1",
          }}
        >
          {images.map((image, index) => {
            return (
              <SwiperSlide key={index} className="w-full h-full">
                <Image
                  src={image.imageURL}
                  alt={image.alt}
                  layout="fill"
                  objectFit="cover"
                />
              </SwiperSlide>
            );
          })}
        </Swiper>
        <button
          className="w-[5%] h-1/2 flex items-center justify-center z-10"
          onClick={() => handleClick("right")}
        >
          <div className="h-fit w-fit">
            <MdChevronRight width={"32px"} height={"32px"} />
          </div>
        </button>
      </div>
    </div>
  );
};
